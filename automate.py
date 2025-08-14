import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "methods", "unidepthv2"))

import subprocess
import sys
from datetime import datetime
from custom_assets.datasets import Dataset
from custom_assets.models import Model, AlignmentType

dtset = Dataset.IPHONE  
model = Model.Torch_UNIDEPTH_V2  

experiments = [
    {
        "name": f"{model.name} on {dtset.name} - No Alignment",
        "dataset": dtset.name,
        "model": model.name,
        "align": False,
        "alignType": AlignmentType.MedianBased.name,
        "alignShift": False,
    },

    {
        "name": f"{model.name} on {dtset.name} - median-based scale alignment",
        "dataset": dtset.name,
        "model": model.name,
        "align": True,
        "alignType": AlignmentType.MedianBased.name,
        "alignShift": False,
    },
    {
        "name": f"{model.name} on {dtset.name} - median-based scale alignment with shift",
        "dataset": dtset.name,
        "model": model.name,
        "align": True,
        "alignType": AlignmentType.MedianBased.name,
        "alignShift": True,
    },

    {
        "name": f"{model.name} on {dtset.name}- fitting-based scale alignment",
        "dataset": dtset.name,
        "model": model.name,
        "align": True,
        "alignType": AlignmentType.Fitting.name,
        "alignShift": False,
    },
    {
        "name": f"{model.name} on {dtset.name}- fitting-based scale alignment with shift",
        "dataset": dtset.name,
        "model": model.name,
        "align": True,
        "alignType": AlignmentType.Fitting.name,
        "alignShift": True,
    },
]

# ==============================================================================
# AUTOMATION SCRIPT LOGIC
# ==============================================================================

start_time = datetime.now()
print(f"Starting experiment suite at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Found {len(experiments)} experiments to run.")
print("==================================================================\n")

for i, config in enumerate(experiments):
    
    print(f"--- Running Experiment {i+1}/{len(experiments)}: {config['name']} ---")
    cmd = [sys.executable, "main.py"]
    cmd.extend(["--dataset", config["dataset"]])
    cmd.extend(["--model", config["model"]])
    cmd.extend(["--alignType", config["alignType"]])
    
    if config["align"]:
        cmd.append("--align")
        
    if config["alignShift"]:
        cmd.append("--alignShift")
    
    print(f"Executing command: {' '.join(cmd[1:])}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"--- Experiment '{config['name']}' completed successfully. ---\n")
        
    except subprocess.CalledProcessError as e:
        print(f"!!! Experiment '{config['name']}' FAILED with return code {e.returncode}. !!!")
        print("Stopping the automation script to prevent further errors.")
        break # Stop running more experiments if one fails
    except FileNotFoundError:
        print("!!! ERROR: 'main.py' not found. Make sure automate.py is in the same directory. !!!")
        break

end_time = datetime.now()
print("==================================================================")
print(f"Experiment suite finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total duration: {end_time - start_time}")