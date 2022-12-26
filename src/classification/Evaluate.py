import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import help_functions as hf
from matplotlib import pyplot as plt
from os.path import exists
from HierarchicalModel import HierarchicalModel

# To run this: `python Evaluation.py 0`
# hf.setup_gpu(gpu_nr=1)

# ================== HYPER-PARAMETERS ==================
# config: nr_classes, labels, class_weights, basemodel, image_dimension, results_and_checkpoints_folder, data_folder
i = sys.argv[1]
with open('training_configurations.json', 'r') as fp:
    config = json.load(fp)[str(i)]
old_stdout = sys.stdout
if not exists(config['results_folder'] + '/log_eval.txt'):
    log_file = open(config['results_folder'] + '/log_eval.txt', 'w')
else:
    raise FileExistsError
sys.stdout = log_file
print('\n\n\n=============== EVALUATION ====================\n')
# ======================================================

# Load test set
test, _ = hf.get_flow(df_file=config['data_folder'] + '/test_df.json.bz2',
                      batch_size=config['batch_size'],
                      nr_classes=config['nr_classes'],
                      image_dimension=config['image_dimension'])

# Create model
if config['hierarchical']:
    model = HierarchicalModel(nr_labels=len(test.class_indices), image_dimension=config['image_dimension'])
else:
    if config['nr_classes'] == 'all':
        n_labels = len(test.class_indices)
    model = hf.create_model(n_labels=n_labels, image_dimension=config['image_dimension'], model_name=config['basemodel'], number_trainable_layers=config['number_trainable_layers'])
# latest = tf.train.latest_checkpoint(config['results_folder'] + '/checkpoints')
latest = tf.train.latest_checkpoint(config['results_folder'])
print(latest)
model.load_weights(latest)

# Predict on test set
print('Predicting on test set:\n')
probs_test = model.predict(test, verbose=2)
y_true_test = hf.get_y_true(shape=(test.samples, len(test.class_indices)), 
                            classes=test.classes)
y_pred_test_05 = 1 * (probs_test > 0.5)

THRESHOLD_MOVE = False
if THRESHOLD_MOVE:
    val_threshold, _ = hf.get_flow(df_file=config['data_folder'] + '/val_threshold_df.json.bz2',
                                   batch_size=config['batch_size'],
                                   nr_classes=config['nr_classes'],
                                   image_dimension=config['image_dimension'])
    # Use the second validation set to the thresholds that optimize f1-score
    print('Predicting on validation set:\n')
    probs_val = model.predict(val_threshold, verbose=2)
    y_true_val = hf.get_y_true(shape=probs_val.shape, classes=val_threshold.classes)

    optim_thresholds = hf.get_optimal_threshold(y_true=y_true_val, 
                                            probs=probs_val, 
                                            thresholds=np.linspace(start=0, stop=1, num=21), 
                                            labels=list(val_threshold.class_indices.keys()), 
                                            N=7,
                                            image_path=config['results_folder'])
    y_pred_test_per_class_threshold = 1 * (probs_test > optim_thresholds)

def plot_f1_scores_side_by_side(f1_scores_05, f1_scores_thresh, image_path):
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(f1_scores_05.keys()))
    plt.bar(x_axis-0.1, f1_scores_05.values, width=0.2, label='Threshold 0.5')
    plt.bar(x_axis+0.1, f1_scores_thresh.values, width=0.2, label='Per-class threshold')
    plt.legend(fontsize=12)
    _ = plt.xticks(x_axis, f1_scores_05.keys(), rotation=70, rotation_mode='anchor', ha="right", fontsize=14)
    # plt.title('F1-scores comparison')
    plt.ylabel('F1-score')
    plt.xlabel('Label')
    try:
        plt.savefig(image_path + '/f1-scores-threshold-moving-or-05.png', bbox_inches='tight')
    except:
        print('Could not save image')


# ================== GET METRICS ======================
print('METRICS FOR THRESHOLD 0.5')
f1_scores_05 = hf.get_metrics(y_true_test, y_pred_test_05, label_names=list(test.class_indices.keys()), image_path=config['results_folder'] + '/f1_scores.png')

if THRESHOLD_MOVE:
    print('METRICS WHEN HAVING PER-CLASS THRESHOLDS')
    f1_scores_thresh = hf.get_metrics(y_true_test, y_pred_test_per_class_threshold, label_names=list(test.class_indices.keys()), image_path=config['results_folder'] + '/f1_scores.png')
    plot_f1_scores_side_by_side(f1_scores_05, f1_scores_thresh, config['results_folder'])

from sklearn.metrics import precision_recall_curve, auc

# precision recall curve
def plot_pr_curve(y_true, probs, label_names):
    plt.figure(figsize=(15,10))
    precision = dict()
    recall = dict()
    pr_auc = dict()
    random_auc = dict()
    n_images = y_true.shape[0]
    n_labels = len(label_names)

    for i in range(n_labels):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        random_auc[i] = y_true[:, i].sum() / n_images

    plt.figure(figsize=(18, 13))
    
    plt.subplot(3, 2, 1)
    for i in range(0, 6):
        plt.plot(recall[i], precision[i], label=f'{label_names[i]}, PR_AUC={round(pr_auc[i], 4)} ({round(pr_auc[i]/random_auc[i], 1)}x random)')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc='upper right')

    plt.subplot(3, 2, 2)
    for i in range(6, 11):
        plt.plot(recall[i], precision[i], label=f'{label_names[i]}, PR_AUC={round(pr_auc[i], 4)} ({round(pr_auc[i]/random_auc[i], 1)}x random)')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc='upper right')

    plt.subplot(3, 2, 3)
    for i in range(12, 18):
        plt.plot(recall[i], precision[i], label=f'{label_names[i]}, PR_AUC={round(pr_auc[i], 4)} ({round(pr_auc[i]/random_auc[i], 1)}x random)')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc='upper right')

    plt.subplot(3, 2, 4)
    for i in range(19, 25):
        plt.plot(recall[i], precision[i], label=f'{label_names[i]}, PR_AUC={round(pr_auc[i], 4)} ({round(pr_auc[i]/random_auc[i], 1)}x random)')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc='upper right')

    plt.subplot(3, 2, 5)
    for i in range(26, 31):
        plt.plot(recall[i], precision[i], label=f'{label_names[i]}, PR_AUC={round(pr_auc[i], 4)} ({round(pr_auc[i]/random_auc[i], 1)}x random)')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend(loc='upper right')
    plt.show()
    hf.save_img(config['results_folder'] + '/pr_curve.png')
    
    return precision, recall, pr_auc, random_auc

precision, recall, pr_auc, random_auc = plot_pr_curve(y_true_test, probs_test, list(test.class_indices.keys()))

metrics_dict = dict()
metrics_dict['label_name'] = list(test.class_indices.keys())
metrics_dict['pr_auc'] = list(pr_auc.values())
metrics_dict['random_auc'] = list(random_auc.values())
metrics_df = pd.DataFrame(metrics_dict)
metrics_df['better_than_random'] = round(metrics_df['pr_auc'] / metrics_df['random_auc'], 3)
macro_pr_auc = metrics_df['pr_auc'].sum() / len(metrics_df)


print(f'\n\n-------------------------- PR AUC metrics table -----------------------')
print(f'Macro PR_AUC is: {macro_pr_auc}')
print(metrics_df)
metrics_df.head(31)

print('\n\n----------------------Sorted by the best PR_AUC-------------------')
print(metrics_df.sort_values('pr_auc', ascending=False))
metrics_df.sort_values('pr_auc', ascending=False)

print('\n\n----------------------Sorted by the best compared to random-------------------')
print(metrics_df.sort_values('better_than_random', ascending=False))
metrics_df.sort_values('better_than_random', ascending=False)