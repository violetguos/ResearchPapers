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
    - Future Direction/Drawbacks
        - Assume hypothesis space is correct. 
    - Related Works
        - Train on a new objective function for maximizing information gain. 
        - Separate objective function from actual objective of minimizing loss for a given goal. 
        - Human designed algorithms have no clear objective functions. Don't know what it's trying to achieve.  
        - Consult entire dataset to select which to be used for training, but consulting time could be expensive. 
        
