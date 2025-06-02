# Chapter 2: Pipeline Task Runner

Welcome back! In [Chapter 1: Configuration Management](01_configuration_management_.md), we learned how the `Field-DataExploration` project gets its instructions by reading settings from configuration files like `conf/config.yaml`. We saw how changing the `pipeline` list in that file lets you decide *which* steps the project should perform.

But knowing *where* the instructions come from is only part of the story. How does the project actually *read* that list of steps and then *do* each one? That's the job of what we call the **Pipeline Task Runner**.

### What is the Pipeline Task Runner?

Imagine you have a checklist for building something. The list tells you:
1.  Get the wood.
2.  Cut the wood.
3.  Screw pieces together.
4.  Paint it.

The **Pipeline Task Runner** is like the person (or robot!) who takes that checklist and goes through it *one item at a time*, making sure each step is completed before moving to the next.

In our project, the "checklist" is the `pipeline` list in `conf/config.yaml` that we saw in the last chapter. The "Pipeline Task Runner" is the main part of the `main.py` script that reads this list and runs the code for each task listed.

**Why do we need this?** Without a task runner, you'd have to manually run each piece of the project code separately, in the correct order. If you wanted to run only steps 1, 3, and 4, you'd have to write a separate script or type commands for each one. The task runner automates this process. It makes running the project easy and flexible – you just update the checklist (the `pipeline` in `config.yaml`) and run `main.py`.

The central use case here is simply **running the project pipeline based on the configuration**. You want to say "run data download, then processing, then reporting" (or any combination!) in `config.yaml`, and have the project automatically execute those steps in that specific order.

### How It Works (The Big Picture)

The Pipeline Task Runner lives inside the `main.py` file. Remember from [Chapter 1: Configuration Management](01_configuration_management_.md) that `main.py` is the starting point.

Here's the basic flow:

1.  You run the `main.py` script.
2.  Hydra (the configuration tool from Chapter 1) jumps into action because of the `@hydra.main` line. It reads `conf/config.yaml` and any other linked config files, putting all the settings into a special object called `cfg`.
3.  Hydra then calls the main function (`run_FIELD_REPORT`) in `main.py`, passing it this `cfg` object.
4.  Inside the `run_FIELD_REPORT` function, the code looks at `cfg.pipeline` to get the list of tasks you defined in the config file.
5.  It then loops through this list, one task name at a time.
6.  For each task name in the list (like `"report"`), it finds the actual Python code responsible for the "report" task.
7.  It then runs that code, making sure to also give *that* code access to the `cfg` object so the task can use any settings it needs (like where to save the report).
8.  Once one task finishes successfully, it moves on to the next task in the list.

Let's visualize this process:

```mermaid
sequenceDiagram
    participant User
    participant MainPy as main.py
    participant HydraLib as Hydra Library
    participant ConfigFiles as Configuration Files
    participant TaskCode as Code for Each Task (e.g., report.py, data_acquisition.py)

    User->MainPy: Run script (python main.py)
    MainPy->HydraLib: Start (via @hydra.main)
    HydraLib->ConfigFiles: Load configuration (conf/config.yaml, etc.)
    ConfigFiles-->HydraLib: Return 'cfg' object
    HydraLib->MainPy: Call run_FIELD_REPORT(cfg)
    MainPy->MainPy: Get task list from cfg.pipeline
    loop For each task in the list
        MainPy->MainPy: Find code for current task (e.g., report.main)
        MainPy->TaskCode: Run task code (passing cfg)
        TaskCode-->MainPy: Task finishes
    end
    MainPy: All tasks completed!
```

This diagram shows how `main.py` uses the configuration (`cfg`) loaded by Hydra to call the individual pieces of code responsible for each part of the pipeline (represented here as `TaskCode`).

### Looking at the Code (main.py)

Now let's look at the core part of `main.py` that makes this happen. We'll simplify it to focus just on the task running logic.

```python
# main.py (Simplified Task Runner part)

import hydra
from omegaconf import DictConfig # Helps Python understand 'cfg' type
from hydra.utils import get_method # Important function!

# This decorator tells Hydra to load config and call this function
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_FIELD_REPORT(cfg: DictConfig) -> None:
    # cfg now holds all your settings from the config files

    # 1. Get the list of tasks from the configuration
    tasks = cfg.pipeline

    # Just logging which tasks are about to run (good practice!)
    # log.info(f"Running {' ,'.join(tasks)}") # Simplified logging

    # 2. Loop through each task name in the list
    for task_name in tasks:
        # 3. Find the actual Python function for this task
        # Example: If task_name is "report", this looks for code in 'report.main'
        task_function = get_method(f"{task_name}.main")

        # 4. Run the found task function, passing the configuration 'cfg'
        try:
            task_function(cfg)
            # If the task finishes without errors, continue the loop
        except Exception as e:
            # If a task fails, log the error and stop the program
            # log.exception("Task failed") # Simplified logging
            pass # In reality, you'd handle errors properly
            # sys.exit(1) # Exit if a task fails

    # 5. Loop finishes when all tasks are done!

# This part is outside the function, runs the decorated function
if __name__ == "__main__":
    run_FIELD_REPORT()
```

Let's break down the key lines inside the `run_FIELD_REPORT` function:

1.  `tasks = cfg.pipeline`: This is where we get the list of task names directly from the `cfg` object provided by Hydra, which contains the values from your `conf/config.yaml` file. If your `config.yaml` listed `["report", "image_inspection"]`, the `tasks` variable would be exactly that list.

2.  `for task_name in tasks:`: This loop iterates through each name in the `tasks` list. In our example `["report", "image_inspection"]`, the loop would run twice: once with `task_name` being `"report"`, and then once with `task_name` being `"image_inspection"`.

3.  `task_function = get_method(f"{task_name}.main")`: This is a neat trick provided by Hydra. `get_method()` takes a string like `"report.main"` and finds the actual Python function named `main` inside a Python module (file) named `report.py`. So, when `task_name` is `"report"`, `get_method` finds the `main` function inside the `src/report.py` file. When `task_name` is `"image_inspection"`, it finds the `main` function inside `src/image_inspection.py`. This is how the project dynamically finds and runs the code for each task listed in the configuration!

4.  `task_function(cfg)`: Once `get_method` finds the correct function, this line simply *calls* that function. Crucially, it passes the `cfg` object to it. This means the `report.py` code, when it runs, receives the same `cfg` object that `main.py` received. It can then look up settings it needs, like `cfg.paths.reportdir` to know where to save the report, or `cfg.inspection.num_past_days_for_report` to know how many days to include, all without having to hardcode these values or figure out how to load them itself.

### How to Use the Task Runner

The best part is, you don't really *do* much directly with the Task Runner code itself in `main.py`. Its job is to read the configuration and execute.

To control *what* the Task Runner does, you go back to [Chapter 1: Configuration Management](01_configuration_management_.md) and modify the `pipeline` list in `conf/config.yaml`.

For example, if `conf/config.yaml` looks like this:

```yaml
# conf/config.yaml (Example)

# ... other settings ...

pipeline:
    - wir_table_generator # Task 1: Generate table data
    - report            # Task 2: Generate reports

# ... other settings ...
```

When you run `python main.py`, the Task Runner will:

1.  Get `tasks = ["wir_table_generator", "report"]`.
2.  Loop 1: Find and run the `main` function in the code for `"wir_table_generator"`, passing it `cfg`.
3.  Loop 2: Find and run the `main` function in the code for `"report"`, passing it `cfg`.

If you change `conf/config.yaml` to:

```yaml
# conf/config.yaml (Modified Example)

# ... other settings ...

pipeline:
    - image_inspection # Task 1: Inspect images
    - report           # Task 2: Generate reports
    - plot_by_season   # Task 3: Generate plots

# ... other settings ...
```

Running `python main.py` again will make the Task Runner:

1.  Get `tasks = ["image_inspection", "report", "plot_by_season"]`.
2.  Loop 1: Find and run code for `"image_inspection"`, passing `cfg`.
3.  Loop 2: Find and run code for `"report"`, passing `cfg`.
4.  Loop 3: Find and run code for `"plot_by_season"`, passing `cfg`.

The Task Runner automatically adapts its behavior based on your configuration!

### Summary

In this chapter, we explored the **Pipeline Task Runner**, the part of the `main.py` script that acts like a project manager executing tasks from a checklist.

We learned that:
*   Its main job is to read the `pipeline` list from the configuration (`cfg.pipeline`).
*   It loops through each task name in that list.
*   Using `hydra.utils.get_method`, it finds the actual Python code (specifically the `main` function in the corresponding module) for each task name.
*   It runs that task's code, passing the full configuration object (`cfg`) to it so the task has access to all necessary settings.
*   This makes the project very flexible – simply change the `pipeline` list in `conf/config.yaml` to run a different set of tasks in a different order.

Now that we understand how the project figures out *which* steps to run and *how* it kicks them off, we're ready to dive into the very first step often performed in the pipeline: getting the data!

Let's move on to the next chapter where we'll learn about **Azure Data Acquisition**.

[Azure Data Acquisition](03_azure_data_acquisition_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)