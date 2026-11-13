import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import geometric_mean

from llmastar.model import ChatGPT, Llama3
from llmastar.pather import AStar, LLMAStar
from llmastar.utils import list_parse

ROOT = Path(__file__).resolve().parent
CONFIDENCE_PROMPT = """
In your final answer, keep the path as coordinate pairs on one line:
Generated Path: [[x1, y1], [x2, y2], ...]
Then add exactly one confidence score per point, in the same order, on a separate line:
Waypoint Confidence: [c1, c2, ...]
Each score must be a number between 0 and 1 expressing your confidence that this
point is a useful, obstacle-avoiding waypoint on a short route to the goal.
Use integer grid coordinates and set the start and goal confidence to 1.
"""


class DynamicWAStar(AStar):
    def __init__(self):
        self.weight = 2.0

    def f_value(self, s):
        return self.g[s] + self.weight * self.heuristic(s)

    def heuristic(self, s):
        return round(super().heuristic(s), 15)

    def get_neighbor(self, s):
        self.weight = max(1.0, self.weight * 0.99)
        self.OPEN = self.updated_queue()
        return super().get_neighbor(s)


class CachedModel:
    """Share the same path and self-reported confidence across all planners/runs."""

    def __init__(self, llm, directory):
        self.llm, self.directory, self.backend = llm, directory, None
        self.cache_hits = 0
        directory.mkdir(parents=True, exist_ok=True)

    def ask(self, prompt, **kwargs):
        prompt += CONFIDENCE_PROMPT
        model_id = "gpt-4o-mini" if self.llm == "gpt" else self.llm
        key = json.dumps([model_id, prompt, kwargs], sort_keys=True)
        path = self.directory / (hashlib.sha256(key.encode()).hexdigest() + ".txt")
        if path.exists():
            self.cache_hits += 1
        else:
            if self.backend is None:
                self.backend = (ChatGPT("LLM-A*", "", None)
                                if self.llm == "gpt" else Llama3())
            response = self.backend.ask(prompt=prompt, **kwargs)
            path.write_text(response, encoding="utf-8")
        self.response = path.read_text(encoding="utf-8")
        return self.response


class Ours(LLMAStar):
    """LLM-A* as sequential subgoal search: an independent search is run for each
    segment (start -> waypoint 1 -> waypoint 2 -> ... -> goal) and the segment paths
    are concatenated. If a waypoint is unreachable, it is skipped and the next one is
    searched from the same anchor. Each segment keeps the original heuristic form
    d(s, goal) + d(s, segment target)."""

    # main() records this attribute; this method uses no waypoint confidence.
    waypoint_confidence = None

    def searching(self, query, filepath='temp.png'):
        self.filepath = filepath
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()

        self._stored = {}
        self.skipped_waypoints = 0
        path, anchor = [self.s_start], self.s_start

        for t in self.target_list[1:]:
            seg = self._segment_search(anchor, t)
            if seg is None:
                if t == self.s_goal:  # goal unreachable from the current anchor
                    path = None
                    break
                self.skipped_waypoints += 1
                continue
            path.extend(seg[1:])
            anchor = t

        self.g = self._stored  # storage: distinct states stored by any segment
        self._path = path

        path = self.extract_path(self.PARENT)
        result = {
            "operation": len(self.CLOSED),
            "storage": len(self.g),
            "length": sum(self._euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1)),
            "llm_output": self.target_list,
        }
        if self.filepath is not None:
            self.plot.animation(path, self.CLOSED, True, "LLM-A*", self.filepath)
        return result

    def _segment_f(self, g, s, target):
        return (g[s] + math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1])
                + math.hypot(target[0] - s[0], target[1] - s[1]))

    def _segment_search(self, origin, target):
        import heapq

        g, parent, closed = {origin: 0}, {origin: origin}, set()
        open_ = [(self._segment_f(g, origin, target), origin)]

        while open_:
            _, s = heapq.heappop(open_)
            closed.add(s)
            self.CLOSED.append(s)  # every segment's expansions are counted

            if s == target:
                break

            for s_n in self.get_neighbor(s):
                if s_n in closed:
                    continue
                new_cost = g[s] + self.cost(s, s_n)
                if s_n not in g:  # keep original storage rule
                    g[s_n] = math.inf
                if new_cost < g[s_n]:
                    g[s_n] = new_cost
                    parent[s_n] = s
                    heapq.heappush(open_, (self._segment_f(g, s_n, target), s_n))

        self._stored.update(g)
        if target not in closed:
            return None
        seg = [target]
        while seg[-1] != origin:
            seg.append(parent[seg[-1]])
        return seg[::-1]

    def extract_path(self, PARENT):
        if self._path is None:  # no path: same KeyError as the original, handled by search()
            raise KeyError(self.s_goal)
        return list(self._path)

def valid_length(nodes, planner):
    """Evaluate the raw LLM polyline without filtering/repairing its nodes."""
    if not isinstance(nodes, list) or len(nodes) < 2:
        return None
    if any(not isinstance(p, (list, tuple)) or len(p) != 2
           or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
           for p in nodes):
        return None
    if tuple(nodes[0]) != planner.s_start or tuple(nodes[-1]) != planner.s_goal:
        return None
    if any(not (planner.range_x[0] < x < planner.range_x[1]
                and planner.range_y[0] < y < planner.range_y[1]) for x, y in nodes):
        return None
    if any(planner.is_collision(a, b) for a, b in zip(nodes, nodes[1:])):
        return None
    return sum(math.dist(a, b) for a, b in zip(nodes, nodes[1:]))


def search(planner, query):
    # Original searching() mutates range_x/range_y: every run needs a fresh copy.
    with redirect_stdout(io.StringIO()):
        try:
            result = planner.searching(deepcopy(query), filepath=None)
        except KeyError as error:
            if error.args != (tuple(query["goal"]),):
                raise
            return {"operation": len(planner.CLOSED), "storage": len(planner.g),
                    "length": None, "valid": False}
    path = planner.extract_path(planner.PARENT)
    if isinstance(planner, AStar):
        path = path[::-1]
    result["valid"] = valid_length(path, planner) is not None
    return result


def percentage(values):
    return round(100 * geometric_mean(values), 2) if values else "-"


def format_table(groups):
    """Append Ours to the paper's layout; leave unmeasured results as dashes."""
    header = ["Methodology", "Base Model", "Prompt Approach", "Operation Ratio ↓ (%)",
              "Storage Ratio ↓ (%)", "Relative Path Length ↓ (%)", "Valid Path Ratio ↑ (%)"]
    method_names = {"Dynamic WA*": "Dynamic WA* (w = 2)"}
    model_names = {"gpt-4o-mini": "GPT-4o mini", "llama": "LLAMA3"}
    prompt_names = {"standard": "Few-Shot", "cot": "CoT", "repe": "RePE"}
    rows = [header, ["---"] * 3 + ["---:"] * 4]
    previous = (None, None)
    ordered = [("A*", "-", "-"), ("Dynamic WA*", "-", "-")]
    ordered += [(method, model, prompt) for method in ("LLM", "LLM-A*", "Ours")
                for model in ("gpt-4o-mini", "llama") for prompt in ("standard", "cot", "repe")]
    for method, model, prompt in ordered:
        records = groups.get((method, model, prompt), [])
        metrics = [percentage([r["ratios"][key] for r in records if key in r["ratios"]])
                   for key in ("operation", "storage", "length")]
        valid = round(100 * sum(r["valid"] for r in records) / len(records), 2) if records else "-"
        cells = [method_names.get(method, method) if method != previous[0] else "",
                 model_names.get(model, model) if (method, model) != previous else "",
                 prompt_names.get(prompt, prompt), *metrics, valid]
        rows.append([v if isinstance(v, str) else str(int(v)) if v == int(v) else f"{v:.2f}"
                     for v in cells])
        previous = (method, model)
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    rows[1] = ["-" * width if i < 3 else "-" * (width - 1) + ":"
               for i, width in enumerate(widths)]
    return "\n".join("| " + " | ".join(cell.ljust(width) if i < 3 else cell.rjust(width)
                                      for i, (cell, width) in enumerate(zip(row, widths))) + " |"
                     for row in rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", choices=["gpt", "llama"], default=["gpt"])
    parser.add_argument("--methods", nargs="+", choices=["dynamic", "llm", "llm-astar", "ours"],
                        default=["dynamic"],
                        help="Methods to run alongside A* (always run); default: dynamic")
    parser.add_argument("--limit", type=int, default=0, help="Number of problems; 0 = all 1000")
    parser.add_argument("--output", type=Path, default=Path("results/table1"))
    parser.add_argument("--cache", type=Path, default=ROOT / "results/cache",
                        help="Shared LLM response cache, independent of --output")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    # 1. Read the original problems.
    dataset = ROOT / "dataset/environment_50_30.json"
    maps = json.loads(dataset.read_text())
    output = args.output.resolve()
    suffix = 0
    while True:
        args.output = output if suffix == 0 else output.with_name(f"{output.name}_{suffix}")
        try:
            args.output.mkdir(parents=True)
            break
        except FileExistsError:
            suffix += 1
    print(f"Output: {args.output}", flush=True)
    samples = [(m, i, sg) for m in maps for i, sg in enumerate(m["start_goal"])]
    samples = samples[:args.limit or len(samples)]
    models = {llm: CachedModel(llm, args.cache / llm) for llm in args.models
              if any(method in args.methods for method in ("llm", "llm-astar", "ours"))}
    groups = {}

    # 2. Run each problem, sharing the cached response across LLM-based methods.
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as log:
        for index, (m, sg_index, sg) in enumerate(samples, 1):
            cache_hits = sum(model.cache_hits for model in models.values())
            query = {**m, "start": sg[0], "goal": sg[1]}
            baseline = search(AStar(), query)
            if not baseline["valid"] or baseline["length"] <= 0:
                raise ValueError(f"Invalid A* reference: map={m['id']}, pair={sg_index}")
            results = [("A*", "-", "-", baseline)]
            if "dynamic" in args.methods:
                results.append(("Dynamic WA*", "-", "-", search(DynamicWAStar(), query)))
            for llm, model in models.items():
                model_name = "gpt-4o-mini" if llm == "gpt" else llm
                for prompt in ("standard", "cot", "repe"):
                    planner = LLMAStar(llm=llm, prompt=prompt, model=model)
                    if "llm-astar" in args.methods:
                        hybrid = search(planner, query)
                        results.append(("LLM-A*", model_name, prompt, hybrid))
                    elif "llm" in args.methods:
                        with redirect_stdout(io.StringIO()):
                            planner._initialize_parameters(deepcopy(query))
                            planner._initialize_llm_paths()
                    if "llm" in args.methods:
                        length = valid_length(list_parse(model.response), planner)
                        results.append(("LLM", model_name, prompt, {"length": length, "valid": length is not None}))
                    if "ours" in args.methods:
                        planner = Ours(llm=llm, prompt=prompt, model=model)
                        result = search(planner, query)
                        result["waypoint_confidence"] = planner.waypoint_confidence
                        results.append(("Ours", model_name, prompt, result))
            for method, llm, prompt, result in results:
                record = {"map": m["id"], "pair": sg_index, "method": method,
                          "model": llm, "prompt": prompt, "reference": "current", **result}
                ratios = {key: result[key] / baseline[key]
                          for key in ("operation", "storage", "length")
                          if key in result and result[key] is not None
                          and (key != "length" or result["valid"])}
                record["ratios"] = ratios
                log.write(json.dumps(record, allow_nan=False) + "\n")
                groups.setdefault((method, llm, prompt), []).append(record)
            log.flush()
            reused = sum(model.cache_hits for model in models.values()) - cache_hits
            status = f" | LLM cache 재사용: {reused}회" if reused else ""
            print(f"{index}/{len(samples)}{status}", flush=True)

    # 3. Aggregate problem-wise ratios and save Table 1.
    table = format_table(groups)
    (args.output / "table1.md").write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()