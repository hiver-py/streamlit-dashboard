# Just some example prompts, change them as you like, or use your favourite model to generate them for you.

default_prompt = "Create a short and professional analysis of a models performance using " \
                 "the provided metrics. Include a simple explanation of each of the performance metrics. " \
                 "Make it as impartial as possible, and don't forget to consider the difference in " \
                 "performance between the different datasets when assessing the reliability of the model." \
                 "An example layout of the report could be to first list the performance metrics, " \
                 "then provide an explanation for each of them, then interpret each metrics individually, and " \
                 "finally make some concluding remarks that evaluates reliability and performance of the model."

metrics_prompt = " These metrics have been obtained by averaging the performance metrics that has been obtained " \
                 "through multiple runs during a models during a hyperparameter search." \
                 "Give a short interpretation of each model metric, and at a level overview provide if there are " \
                 "reasons for concern regarding the models reliability, generalization and use case by taking the " \
                 "training and test metrics in regard, as well as the difference between the " \
                 "train and test performance."


metrics_difference_prompt = "These metrics have been obtained by computing the difference in train and test performance, " \
                            "then averaging this difference that has been obtained through multiple runs during a models" \
                            "during a hyperparameter search. Interpret if given the difference in the performance metrics " \
                            "there is a reason for concern of overfitting. Keep the text short. "

