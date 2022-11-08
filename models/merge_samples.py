import argparse
import pickle
import torch
import glob

if __name__ == '__main__':
    # Get hyperparams from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()

    # Find all the files with the given prefix
    filenames = glob.glob(f"{args.prefix}*")

    # Merge them together
    assert len(filenames) >= 1
    #   Use the first filename as a "base"
    with open(filenames[0], "rb") as filehandle:
        num_samples, SL_dataset = pickle.load(filehandle)
    #   Open the rest and merge into the "base"
    for filename in filenames[1:]:
        with open(filename, "rb") as filehandle:
            num_samples_new, SL_dataset_new = pickle.load(filehandle)

        # Merge
        for partition in SL_dataset:
            for idx in range(len(SL_dataset[partition])):
                Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset[partition][idx]
                Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset_new[partition][idx]

                # Sanity check
                assert torch.isclose(Y_old, Y_new).all()

                # Combine entries
                opt_objective = max(opt_objective_new, opt_objective_old)
                Yhats = torch.cat((Yhats_old, Yhats_new), dim=0)
                objectives = torch.cat((objectives_old, objectives_new), dim=0)

                # Update
                SL_dataset[partition][idx] = (Y_old, opt_objective, Yhats, objectives)
        num_samples += num_samples_new

    # Write to new file
    with open(f"{args.prefix}.pkl", 'wb') as filehandle:
        pickle.dump((num_samples, SL_dataset), filehandle)