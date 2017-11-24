# Active Learning
//----------------------------------------------------------------------------------------------------
FORMAT
TitleOfTopicThatThesePapersAreAbout
- Title of Paper followed by link to paper https://arxiv.org/pdf/1407.2806v1.pdf 
    - Motivation/Problem:
        - The problem or motivation that the paper was written
    -Contributions/Results (math proofs/experiments/newDataSet):
        - Issues with matrix factorization with cold start. 
        - What was contributed by this paper
    - Future Direction/Drawbacks
        - What are the future directions of this paper for the author
        - Drawbacks of the paper
        - Assumptions Made
        - (Optional) Method/Approach Used
    - Brief discussion of the algorithm used if it's obvious to understand. 
        - (Optional) Related work information that may be useful
        - (Optional) Datasets used so we know if we are able to use them
        - (Optional) Anything else you want to add about the paper as summarize notes. 
- Title of Next paper followed by link to that paper https://arxiv.org/pdf/1407.2806v1.pdf 
//----------------------------------------------------------------------------------------------------


## Active Learning
- [Information-Based Objective Functions for Active Data Selection](https://authors.library.caltech.edu/13795/1/MACnc92c.pdf)
    - Motivation/Problem:
        - Each data can be expensive, for instance, self driving car accidents. 
        - Too much data, no time to train on all data. 
        - Therefore, want to select which data to use for learning. 
    - Contribution
        - Select salient data points during learning using objective function that measure expected amount of information. 
        - 3 types of information to gather.
            - Maximal information for the weights of a given model. 
            - Predict value accurately in limited region instead of globally. 
            - Maximal information to discriminate between different models. 
        - Pick datapoint that maximizes the entropy of the difference between weight distribution
          before datapoint and weight distribution after datapoint. 
        - Prove that expectation of this approach is same as expectation of cross entropy between both distributions.
        - Generalizes to multiple data point and multiple outputs case. 
    - Future Direction/Drawbacks
        - Assume hypothesis space is correct. 
    - Related Works
        - Train on a new objective function for maximizing information gain. 
        - Separate objective function from actual objective of minimizing loss for a given goal. 
        - Human designed algorithms have no clear objective functions. Don't know what it's trying to achieve.  
        - Consult entire dataset to select which to be used for training, but consulting time could be expensive. 
        
## Curriculum Learning
- [Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003)
    - Motivation/Problem: 
        - Learn easy problems first, then learn harder problems. 
        - However, assume that the difficulty can be ordered, which can be wrong since:
            - difficulty can be in multiple dimensions
            - there is no order of difficulty at all. 
        - Also, need to tune random hyperparameters to learn:
            - when to move to next difficulty
            - re-learn basics to prevent forgetting. 
        - Learning progress are used as reward signals to encourage explorations. 
    - Contribution
        - Focuses on Prediction Gain, a type of indicator progress. 
    - Future Direction/Drawbacks 
    - Related work: 
        - Indicators of learning progress are: 
            - compression progress
            - information acquisition
            - Bayesian surprise
            - prediction gain
            - variational information maximisation. 
        - Learning progress is the rate at which you minimize loss. 
        - Learning progress signals can be: 
            - Loss driven => Lower loss means higher progress
                - Prediction gain
                - Gradient prediction gain
                - Self prediction gain
                - Target prediction gain
                - Mean prediction gain
            - Complexity driven => Higher complexity means higher progress. 
                - Variational Complexity Gain
                - Gradient variational complexity gain
                - Variational Information Maximizing Exploration
                - L2 Gain

## Bandits Evaluation
- [Data-driven evaluation of Contextual Bandit algorithms and applications to Dynamic Recommendation](https://hal.archives-ouvertes.fr/tel-01297407/document)
