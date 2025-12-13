# Behavioral Comparison Evaluation

Compares models against human behavior by running matched environment episodes using episode seeds.

Contains evaluate_progression_recurrent() which runs model on human episodes and computes percentage difference in returns: (model_return - human_return) / |human_return| * 100. Provides scale-invariant behavioral similarity measure. Supports optional CL features (session/run IDs) appended to observations. Returns mean/std of percentage differences and raw model returns. Used by dl/optim/sgd.py and ne/optim/*.py during training.
