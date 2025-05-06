import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import openai
from openai import AzureOpenAI
import wandb

model = "gpt-4"

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_VERSION = "2024-08-01-preview"

def init_archive(config):
    fitnesses = [
        7.887500,  # DPO
        7.881250,  # HINGE
        7.84,  # IPO
        7.603125,  # KTO
    ]
    archive = []
    archive.append(
        {  # DPO
            "code": """
def logistic_log_loss(
    self,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> torch.FloatTensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(self.beta * logits)
    return losses
    """,
            "fitness": None,
        }
    )
    archive.append(
        {  # HINGE
            "code": """
def hinge_loss(
    self,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> torch.FloatTensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = torch.relu(1 - self.beta * logits)
    return losses
    """,
            "fitness": None,
        }
    )
    archive.append(
        {  # IPO
            "code": """
def ipo_loss(
    self,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> torch.FloatTensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = (logits - 1 / (2 * self.beta)) ** 2
    return losses
    """,
            "fitness": None,
        }
    )
    archive.append(
        {  # KTO
            "code": """
def kto_pair_loss(
    self,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> torch.FloatTensor:
    chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
    rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
    losses = torch.cat(
        (
            1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
            1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
        ),
        0,
    )
    return losses
    """,
            "fitness": None,
        }
    )

    for archive_entry, new_fitness in zip(archive, fitnesses):
        archive_entry["fitness"] = new_fitness

    return archive

def validate_code(code: str) -> bool:
    # Run code through test
    try:
        # Namespace dictionary to hold the execution context
        namespace = {}

        # Execute the function definition string within the provided namespace
        exec(code, globals(), namespace)

        names = list(namespace.keys())
        if len(names) != 1:
            return False, f"{len(names)} things in namespace. Please only provide 1"
        func = namespace[names[0]]
        if not callable(func):
            return False, f"{func} is not callable"

        # Create a class to hold the sigmoid_loss function
        class LossModel:
            def __init__(self, beta):
                self.beta = beta

        # Add the function to the class
        setattr(LossModel, "epo_loss", func)

        model = LossModel(beta=0.05)

        # Define input tensors with requires_grad to check gradients
        policy_chosen_logps = torch.randn(10, requires_grad=True)
        policy_rejected_logps = torch.randn(10, requires_grad=True)
        reference_chosen_logps = torch.randn(10, requires_grad=True)
        reference_rejected_logps = torch.randn(10, requires_grad=True)

        # Compute the loss
        loss = model.epo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # Check for NaNs in the output
        if torch.isnan(loss).any():
            return False, "Loss contains NaNs"

        # Check the shape of the output
        if loss.shape != (10,):
            return (
                False,
                f"Expected loss shape to be per input (e.g. (10,)), got {loss.shape}",
            )

        # Backward pass to compute gradients
        loss.mean().backward()

        # Check for NaNs in gradients
        for param in [
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        ]:
            if torch.isnan(param.grad).any():
                return False, f"Gradient for {param} contains NaNs"

        return True, ""

    except Exception as e:
        return False, str(e)


def train_gpo(info):
    # STORE CODE
    with open(f"recipes/esm3/gpo/tests.json", "r") as f:
        tests = json.load(f)
    info["name"] = info["name"].lower().replace(" ", "_")
    for test in tests:  # MAKE NAME UNIQUE.
        if test["name"] == info["name"]:
            info["name"] = f"{info['name']}_{len(tests)}"

    tests.append(
        {
            "name": info["name"],
            "code": info["code"],
        }
    )
    with open(f"recipes/esm3/gpo/tests.json", "w") as f:
        json.dump(tests, f, indent=4)
    # Define the command as a list of arguments
    command = [
        "accelerate",
        "launch",
        "--config_file",
        "recipes/accelerate_configs/deepspeed_zero3.yaml",
        "scripts/run_gpo.py",
        f"recipes/esm3/gpo/config_full.yaml",
    ]

    # Set environment variables directly in the subprocess call
    env = dict(
        os.environ, ACCELERATE_LOG_LEVEL="info"
    )  # Copy current environment and add/modify

    # Execute the command
    result = subprocess.run(command, env=env)
    if result.returncode == 0:
        return True, ""
    return False, f"Failed with return code: {result.returncode}\n{result.stderr}"

def evaluate_gpo(info, config):
    model_score = 0
    return True, model_score

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--num-generations", type=int, default=5)
parser.add_argument("--do-baselines", action="store_true", default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    now = str(datetime.now()).replace(" ", "_")
    
    save_dir = f"runs/{now}"
    os.mkdir(save_dir)

    config = {
        "NUM_GENERATIONS": args.num_generations,
        "SAVE_DIR": save_dir,
        "NOW": now,
    }
    # TODO: tell it to optimize proteins
    system_prompt = """
You are a machine learning researcher who is testing out different RLHF loss functions. When you respond, output a JSON where the first key ("thought") corresponds to your thought process when designing the next function. The second key ("name") corresponds to the name of your next function. Finally, the last key ("code") corresponds to the exact python code that you would like to try. Here is an example:

{"thought": "Based on the previous outputs, I should try the direct preference optimization algorithm.",
"name": "dpo",
"code": "def sigmoid_loss(
    self,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> torch.FloatTensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(self.beta * logits)
    return losses"
}

You are deeply familiar with binary classification losses from the literature. Be creative and reference prior literature when possible.

You must use the exact function interface used above. Feel free to define extra hyperparameters within your function as constants. Do not make them attributes of self.

Note that `self.beta = 0.05`.

RLHF loss functions train on a dataset of pairs of preferred and rejected completions.
`policy_chosen_logps` refers to the policy's log probabilities of the preferred completion, and `policy_rejected_logps` refers to the policy's log probabilities of the rejected completion.
`reference_chosen_logps` and `reference_rejected_logps` refer to the same for the reference (base) model.

The user will then return to you a fitness that corresponds to the performance of the resulting model on a downstream task. Your goal is to maximize performance.
"""
    archive = init_archive(config)
    first_prompt = f"""
Here are some results we've obtained: \n{archive}

Please generate the next one.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_prompt},
    ]

    if args.wandb:
        run = wandb.init(
            project="gbl_project",
            name=args.run_name,
            config=config,
        )
        
    if args.do_baselines:
        baseline_names = ["dpo", "hinge", "ipo", "kto"]
        for archive_entry, name in zip(archive, baseline_names):
            archive_entry["name"] = f"bline-{name}-{args.group}"
            
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=API_VERSION
    )
            
            
    t0 = time.time()
    t_start, t_completion, t_train_start, t_train_end, t_eval_end = t0, t0, t0, t0, t0
    for i in range(args.num_generations):
        if i > 0 and args.wandb:
            columns = ["generation", "thought", "function", "fitness", "next prompt"]
            log_table = wandb.Table(columns=columns)
            if "thought" in out and "code" in out:
                log_table.add_data(i, out["thought"], out["code"], fitness, next_prompt)
            else:
                log_table.add_data(i, "", "", fitness, next_prompt)
            wandb.log(
                {
                    "fitness": fitness,
                    "generation": i,
                    "table": log_table,
                    "generation_time": time.time() - t_start,
                    "code_time": t_completion - t_start,
                    "train_time": t_train_end - t_train_start,
                    "eval_time": t_eval_end - t_train_end,
                }
            )
        t_start = time.time()
        # GENERATE CODE
        if not args.do_baselines:
            for _ in range(API_MAX_RETRY):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=2048,
                        n=1,
                        response_format={"type": "json_object"},
                    ).choices[0]
                    break
                except openai.error.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(API_RETRY_SLEEP)
                except openai.error.InvalidRequestError as e:
                    print(type(e), e)
                    break
            t_completion = time.time()

            messages.append(completion.message.to_dict())
            with open(f"{save_dir}/messages.json", "w") as f:
                json.dump(messages, f, indent=4)
            out = json.loads(completion.message.content)
        else:
            out = {"name": archive[i]["name"], "code": archive[i]["code"]}

        # VALIDATE CODE
        valid, error = validate_code(out["code"])
        if not valid:
            next_prompt = (
                f"Code not valid. Error:\n{error}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            print("CODE NOT VALID")
            continue
        t_train_start = time.time()

        # TRAIN GPO
        trained, error = train_gpo(out, config)
        if not trained:
            next_prompt = (
                f"Training failed. Error:\n{error}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            print("FAILED TRAINING")
            continue
        t_train_end = time.time()

        # EVALUATE GPO
        evaluated, val = evaluate_gpo(out, config)
        if not evaluated:
            next_prompt = (
                f"Evaluation failed. Error:\n{val}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            print("FAILED EVAL")
            continue
        t_eval_end = time.time()

        next_prompt = f"Fitness: {val}.\nPlease generate the next one."
        messages.append({"role": "user", "content": next_prompt})
        fitness = val
