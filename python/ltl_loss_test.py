import torch

from loss_executables import make_input, run_exp, parse_output

def make_sat_data(steps):
    def sample(k):
        return torch.tensor([k, k + 1.0])

    ts = tuple([sample(k) for k in range(1, steps + 1)])
    return torch.stack(ts)

mode = "both"

def experiment(label, executable, constr, data):
    print(label + " (loss, gradient tensor):")
    result = run_exp(executable, "both", 0.01, constr, data)
    if result.returncode != 0:
        print("Error!")
        print(result.stderr)

    (loss, gradT) = parse_output(list(data.shape), result.stdout)
    print(loss)
    print(gradT)

def main():
    # LTL constraint: first state always less than second
    constraint = "G 0 < 1"

    # Data: 10 time steps of trivially satisfying data
    data = make_sat_data(10)

    # Print inputs being used
    print("Constraint: " + constraint)
    print("Data:")
    print(data)
    print()

    # Run experiments with gamma 0.01 for all targets
    experiment("OCaml", "ltl-loss-ocaml", constraint, data)

if __name__ == "__main__":
    main()
