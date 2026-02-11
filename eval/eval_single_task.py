from eval.task_resolver import resolve_one_task, discover_tasks
from eval.judge_runner import judge_one_task
from eval.score_task import score_one_task, write_run_report
from dataclasses import asdict

run_name = "veo_output_test_1"

run_dir = f"runs/{run_name}"
tasks_root = "benchmark/tasks/"
task_map = discover_tasks(tasks_root)

outputs_json_path = f"outputs/{run_name}/veo_output_map_test_1.json"
task_id = "lightbulb_and_switch_01"
video_link = "lightbulb_and_switch_01.mp4"

task = resolve_one_task(
    task_id=task_id, 
    video_link=video_link, 
    task_map=task_map, 
    run_dir=run_dir,
    outputs_json_path=outputs_json_path
)

judge_result = judge_one_task(task, provider='gemini', model='gemini-3-pro-preview')

task_score = score_one_task(judge_result)
_ = write_run_report(task_scores={task_id: task_score}, run_dir=run_dir)