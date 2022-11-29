'''
Pretraining Tasks -- 3 Prompt Families (A, B, C)
'''

all_tasks = {}


# =====================================================
# Task Subgroup A -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_A = {}

template = {}

'''
Input template:
Given the following purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
predict next possible item to be purchased by the user?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-1"

task_subgroup_A["A-1"] = template


template = {}
'''
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Can you help me decide?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the purchase history list of user_{} : \n {} \n I wonder what is the next item to recommend to the user . Can you help me decide ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-2"

task_subgroup_A["A-2"] = template


template = {}
'''
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
try to recommend next item to the user
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history list of user_{} : \n {} \n try to recommend next item to the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-3"

task_subgroup_A["A-3"] = template


template = {}

'''
Input template:
Given the following purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
predict next possible item for the user
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following purchase history of {} : \n {} \n predict next possible item for the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-4"

task_subgroup_A["A-4"] = template


template = {}
'''
Input template:
Based on the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you decide the next item likely to be purchased by the user?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the purchase history of {} : \n {} \n Can you decide the next item likely to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-5"

task_subgroup_A["A-5"] = template


template = {}
'''
Input template:
Here is the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user?
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history of {} : \n {} \n What to recommend next for the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-6"

task_subgroup_A["A-6"] = template


# Pairwise Prediction
template = {}
'''
Input template:
User {{user_id}} has the following purchase history:
{{history item list of {{item_id}}}}
Does the user likely to buy {{item [item_id]}} {{item_photo}} next?
 
Target template:
{{answer_choices[label]}} (yes/no)
 
Metrics:
Accuracy
'''
template['source'] = "user_{} has the following purchase history : \n {} \n does the user likely to buy {} {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'purchase_history', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "A-7"

task_subgroup_A["A-7"] = template


template = {}
'''
Input template:
According to {{user_desc}}'s purchase history list:
{{history item list of {{item_id}}}}
Predict whether the user will purchase {{item [item_id]}} {{item_photo}} next?
 
Target template:
{{answer_choices[label]}} (yes/no)
 
Metrics:
Accuracy
'''
template['source'] = "According to {} 's purchase history list : \n {} \n Predict whether the user will purchase {} {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 4
template['source_argv'] = ['user_desc', 'purchase_history', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "A-8"

task_subgroup_A["A-8"] = template


template = {}
'''
Input template:
According to the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you recommend the next possible item to the user?
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "According to the purchase history of {} : \n {} \n Can you recommend the next possible item to the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "A-9"

task_subgroup_A["A-9"] = template


all_tasks['sequential'] =  task_subgroup_A


# =====================================================
# Task Subgroup B -- Direct -- 8 Prompts
# =====================================================

task_subgroup_B = {}

template = {}

'''
Input template:
Will user {{user_id}} likely to interact with item {{item_id}} {{item_photo}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "Will user_{} likely to interact with item_{} {} ?"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "B-1"

task_subgroup_B["B-1"] = template


template = {}

'''
Input template:
Shall we recommend item {{item_id}} {{item_photo}} to {{user_desc}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "Shall we recommend item_{} {} to {} ?"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 3
template['source_argv'] = ['item_id', 'item_photo', 'user_desc']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "B-2"

task_subgroup_B["B-2"] = template


template = {}

'''
Input template:
For {{user_desc}}, do you think it is good to recommend {{item_title}} {{item_photo}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "For {}, do you think it is good to recommend {} {} ?"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "B-3"

task_subgroup_B["B-3"] = template


template = {}

'''
Input template:
I would like to recommend some items for user {{user_id}}. Is the following item a good choice?
{{item_title}} {{item_photo}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "I would like to recommend some items for user_{} . Is the following item a good choice ? \n {} {}"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "B-4"

task_subgroup_B["B-4"] = template


template = {}

'''
Input template:
Which item of the following to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Which item of the following to recommend for {} ? \n {}"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "B-5"

task_subgroup_B["B-5"] = template


template = {}

'''
Input template:
Choose the best item from the candidates to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Choose the best item from the candidates to recommend for {} ? \n {}"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "B-6"

task_subgroup_B["B-6"] = template


template = {}

'''
Input template:
Pick the most suitable item from the following list and recommend to user {{user_id}}:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Pick the most suitable item from the following list and recommend to user_{} : \n {}"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "B-7"

task_subgroup_B["B-7"] = template


template = {}

'''
Input template:
We want to make recommendation for user {{user_id}}. Select the best item from these candidates:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "We want to make recommendation for user_{} .  Select the best item from these candidates : \n {}"
template['target'] = "{}"
template['task'] = "direct"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "B-8"

task_subgroup_B["B-8"] = template


all_tasks['direct'] = task_subgroup_B



# ====================================================
# Task Subgroup C -- Explanation -- 12 Prompts
# ====================================================

task_subgroup_C = {}

template = {}

'''
Input template:
Generate an explanation for user {{user_id}} about this product: {{item_title}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Generate an explanation for user_{} about this product : {} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-1"

task_subgroup_C["C-1"] = template


template = {}
'''
Input template:
Given the following review headline 
{{review_headline}}
can you help generate an explanation of user {{user_id}} for item {{item_id}} {{item_photo}}?


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''
template['source'] = "Given the following review headline \n {} \n can you help generate an explanation of user_{} for item_{} {} ?"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['review_headline', 'user_id', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-2"

task_subgroup_C["C-2"] = template


template = {}
'''
Input template:
Help user {{user_id}} generate a {{star_rating}}-star explanation about this product: 
{{item_title}} {{item_photo}}
 
 
Target template:
{{explanation}}
 
 
Metrics:
BLUE, ROUGE
'''
template['source'] = "Help user_{} generate a {}-star explanation about this product : \n {} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'star_rating', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-3"

task_subgroup_C["C-3"] = template


template = {}

'''
Input template:
Generate an explanation for {{user_desc}} about this product: {{item_title}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Generate an explanation for {} about this product : {} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-4"

task_subgroup_C["C-4"] = template


template = {}
'''
Input template:
Based on the following review headline:
{{review_headline}}
Generate {{user_desc}}'s purchase explanation about {{item_title}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''
template['source'] = "Based on the following review headline : \n {} \n Generate {} 's purchase explanation about {} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['review_headline', 'user_desc', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-5"

task_subgroup_C["C-5"] = template


template = {}
'''
Input template:
Help {{user_desc}} generate a {{star_rating}}-star explanation for item {{item_id}} {{item_photo}}
 
 
Target template:
{{explanation}}
 
 
Metrics:
BLUE, ROUGE
'''
template['source'] = "Help {} generate a {}-star explanation for item_{} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['user_desc', 'star_rating', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-6"

task_subgroup_C["C-6"] = template


template = {}

'''
Input template:
Predict the star rating, then use {{feature}} as feature word to generate user {{user_id}} 's purchase explanation for item {{item_id}} {{item_photo}}


Target template:
{{star_rating}}, {{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Predict the star rating , then use {} as feature word to generate user_{} 's purchase explanation for item_{} {}"
template['target'] = "{} , {}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['feature', 'user_id', 'item_id', 'item_photo']
template['target_argc'] = 2
template['target_argv'] = ['star_rating', 'explanation']
template['id'] = "C-7"

task_subgroup_C["C-7"] = template


template = {}

'''
Input template:
What score will {{user_desc}} rate item {{item_id}} {{item_photo}}? Then give an explanation for the rating score. (1 being lowest and 5 being highest)


Target template:
{{star_rating}}, {{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "What score will {} rate item_{} {} ? Then give an explanation for the rating score . ( 1 being lowest and 5 being highest )"
template['target'] = "{} , {}"
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'item_id', 'item_photo']
template['target_argc'] = 2
template['target_argv'] = ['star_rating', 'explanation']
template['id'] = "C-8"

task_subgroup_C["C-8"] = template


template = {}
'''
Name:
Input template:
Based on the feature word {{feature}}, generate an explanation for user {{user_id}} about this product: {{item_title}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Based on the feature word {} , generate an explanation for user_{} about this product : {} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['feature', 'user_id', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-9"

task_subgroup_C["C-9"] = template


template = {}
'''
Input template:

Given the word {{feature}}, can you help generate an explanation for {{user_desc}} about the product: \n {{item_title}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Given the word {} , can you help generate an explanation for {} about the product : \n {} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 4
template['source_argv'] = ['feature', 'user_desc', 'item_title', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-10"

task_subgroup_C["C-10"] = template


template = {}
'''
Name: 
Input template:
Using the word {{feature}}, write a {{star_rating}}-star explanation for user {{user_id}} about item {{item_id}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "Using the word {} , write a {}-star explanation for user_{} about item_{} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 5
template['source_argv'] = ['feature', 'star_rating', 'user_id', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-11"

task_subgroup_C["C-11"] = template


template = {}
'''
Name:
Input template:
According to the feature word {{feature}}, generate a {{star_rating}}-star explanation for {{user_desc}} about item {{item_id}} {{item_photo}}


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
'''

template['source'] = "According to the feature word {} , generate a {}-star explanation for {} about item_{} {}"
template['target'] = "{}"
template['task'] = "explanation"
template['source_argc'] = 5
template['source_argv'] = ['feature', 'star_rating', 'user_desc', 'item_id', 'item_photo']
template['target_argc'] = 1
template['target_argv'] = ['explanation']
template['id'] = "C-12"

task_subgroup_C["C-12"] = template


all_tasks['explanation'] = task_subgroup_C
