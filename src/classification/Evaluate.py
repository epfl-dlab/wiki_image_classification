import pandas as pd
import numpy as np
import tensorflow as tf
import json
import sys
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from help_functions import create_model, get_top_classes

# To run this: `python Evaluation.py 0`
tf.setup_gpu(gpu_nr=1)

# ================== HYPER-PARAMETERS ==================
# config: nr_classes, labels, class_weights, basemodel, image_dimension, results_and_checkpoints_folder, data_folder
i = sys.argv[1]
with open('training_configurations.json', 'r') as fp:
    config = json.load(fp)[str(i)]
old_stdout = sys.stdout
log_file = open(config['results_and_checkpoints_folder'] + '/log_eval.txt', 'w')
sys.stdout = log_file
print('\n\n\n=============== EVALUATION ====================\n')
# ======================================================


# ====================== LOAD TEST SET =================
test_df = pd.read_json(config['data_folder'] + '/test_df.json.bz2', compression='bz2')
train_df = pd.read_json(config['data_folder'] + '/train_df.json.bz2', compression='bz2')
top_classes = get_top_classes(config['nr_classes'], train_df) # OBS: are they always the same as top classes of train_df? In the 10-case yes.
# Only keep rows which have either of the top classes
ids_x_labels = test_df.labels.apply(lambda classes_list: any([True for a_class in top_classes if a_class in classes_list]))
test_set_x_labels = test_df[ids_x_labels]
test_set_x_labels['labels'] = test_df['labels'].apply(lambda labels_list: [label for label in labels_list if label in top_classes])
test_df = test_set_x_labels.copy()

datagen = ImageDataGenerator() 
test = datagen.flow_from_dataframe(
        dataframe=test_df, 
        directory='/scratch/WIT_Dataset/images',
        color_mode='rgb',
        x_col='url', 
        y_col='labels', 
        class_mode='categorical', 
        target_size=(config['image_dimension'], config['image_dimension']),
        shuffle=False
        )

N_LABELS = len(test.class_indices)
# ======================================================




# ============== CREATE MODEL, LOAD WEIGHTS ============
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def create_model(name):
    if name == 'EfficientNetB0':
        efficient_net = EfficientNetB0(include_top=False, weights='imagenet', classes=config['nr_classes'],
                                           input_shape=(64, 64, 3))
    elif name == 'EfficientNetB1':
        efficient_net = EfficientNetB1(include_top=False, weights='imagenet', classes=config['nr_classes'],
                                           input_shape=(config['image_dimension'], config['image_dimension'], 3))
    elif name == 'EfficientNetB2':
        efficient_net = EfficientNetB2(include_top=False, weights='imagenet', classes=config['nr_classes'],
                                           input_shape=(config['image_dimension'], config['image_dimension'], 3))

    # efficient_net.trainable=False

    model = Sequential([
        efficient_net,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(config['nr_classes'], activation='sigmoid')
    ])
    # compile() configures the model for training, not necessary when only evaluating
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', 'categorical_accuracy'])

    # model.summary()
    return model
model = create_model(name=config['basemodel'])
latest = tf.train.latest_checkpoint(config['results_and_checkpoints_folder'])
model.load_weights(latest)
# ======================================================





# =============== PREDICT ON TEST SET ==================
from sklearn.metrics import classification_report

predictions = model.predict(test, verbose=2)
threshold = 0.5
y_pred = 1 * (predictions > threshold)
y_pred = predictions
y_true = np.zeros(y_pred.shape)
for row_idx, row in enumerate(test.classes):
    for idx in row:
        y_true[row_idx, idx] = 1
# np.save(file=config['results_and_checkpoints_folder'] + '/y_pred', arr=predictions)
# np.save(file=config['results_and_checkpoints_folder'] + '/y_true', arr=y_true)
# ======================================================



# ========== PLOT PREDICTIONS AND THRESHOLDS ==========
# Inspiration from https://www.kaggle.com/code/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb/notebook
fig, axs = plt.subplots(5, 4, figsize=(12, 12))
fig.tight_layout(h_pad=2.0)
nr_classes = predictions.shape[1]
labels = list(test.class_indices.keys())
bins = np.linspace(0, 1, 75)
for i, ax, label in zip(range(nr_classes), axs.flatten(), labels):
    ax.hist(predictions[y_true[:, i] == 0][:, i], bins, alpha=0.5, label='false', log=True)
    ax.hist(predictions[y_true[:, i] == 1][:, i], bins, alpha=0.5, label='true', log=True)
    # plt.axvline(x=thresholds[i], color='k', linestyle='--') # TODO upload thresholds from training!!!
    ax.legend(loc='upper right')
    ax.set_title(label)
plt.savefig(config['results_and_checkpoints_folder'] + '/prediction_thresholds.png')
# =====================================================



# ============== CONFUSION MATRICES ===================
from sklearn.metrics import multilabel_confusion_matrix
confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
import seaborn as sns
from matplotlib.colors import LogNorm

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    # From https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, cmap='YlGnBu', norm=LogNorm())
    except ValueError:
        raise ValueError('Confusion matrix values must be integers.')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label', fontsize=8)
    axes.set_xlabel('Predicted label', fontsize=8)
    axes.set_title(class_label)

fig, ax = plt.subplots(5, 4, figsize=(10, 10))
    
for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, list(test.class_indices.keys())):
    print_confusion_matrix(cfs_matrix, axes, label, ['N', 'Y'])
    
fig.tight_layout()
plt.savefig(config['results_and_checkpoints_folder'] + '/confusion_matrix.png')
                                                     
# Extra:
print(f'\nMean number of label assignments per image in ground-truth: {np.sum(y_true) / y_true.shape[0]:.4f}')
print(f'Mean number of label assignments per image in predictions: {np.sum(y_pred) / y_pred.shape[0]:.4f}\n')
                                                     
# ======================================================

                                                     
                                                     
# ================== GET METRICS ======================

metrics_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=list(test.class_indices), output_dict=True)).transpose()
metrics_df['index'] = np.concatenate((np.arange(start=0, stop=N_LABELS), [None, None, None, None]))
print(metrics_df)

# F1-score
sorted_indices_f1score = np.argsort(metrics_df['f1-score'][0:N_LABELS])
sorted_f1score_per_class = metrics_df['f1-score'][0:N_LABELS][sorted_indices_f1score]

print(f'\nUnweighted avg. F1-score of all classes: {np.sum(sorted_f1score_per_class) / len(sorted_f1score_per_class)}')
print(f'Unweighted avg. F1-score of top 5 classes: {np.sum(sorted_f1score_per_class[-4:]) / 5}')
print(f'Unweighted avg. F1-score of the rest: {np.sum(sorted_f1score_per_class[0:-4]) / (len(sorted_f1score_per_class) - 5)}\n')

_ = plt.figure(figsize=(8, 14))
                
_ = plt.title('F1-score per class')
_ = plt.barh(range(y_true.shape[1]), sorted_f1score_per_class, color='blue', alpha=0.6)
_ = plt.yticks(ticks=range(N_LABELS), labels=np.array(list(test.class_indices.keys()))[sorted_indices_f1score])
_ = plt.xlabel('F1-score')
_ = plt.grid(True)

plt.savefig(config['results_and_checkpoints_folder'] + '/f1_scores.png')

# Per-class accuracy
print('\n------- Per-class accuracy: --------\n')
from collections import Counter
total = Counter()
correct = Counter()
for i in range(len(test.classes)):
    true_y = test.classes[i]
    for l in true_y:
        total[l]+=1
    predicted_y = np.argwhere(predictions[i] >= 0.5)
    for p in predicted_y:
        if p[0] in true_y:
            correct[p[0]]+=1

name_id_map = test.class_indices
class_names = len(name_id_map)*[0]
for k in name_id_map.keys():
    class_names[name_id_map[k]] = k
            
for k in sorted(total.keys()):
    print(class_names[k].split(".")[-1], "{}/{} == {}".format(correct[k], total[k], round(correct[k]/total[k], 3)))
# ======================================================





# ================== ROC CURVE PER CLASS ===============
from sklearn.metrics import roc_curve, auc
from itertools import cycle

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(N_LABELS):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
lw = 2
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_LABELS)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(N_LABELS):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= N_LABELS

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves

plt.figure(figsize=(12, 8))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average: { roc_auc['micro'] :0.2f}",
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average: { roc_auc['macro'] :0.2f}",
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue", "forestgreen", "black", "red", "yellow", "peru", "olive", "lawngreen", "slategray"])
for i, color in zip(range(N_LABELS), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label=f"{list(name_id_map.keys())[i]}: {roc_auc[i]:0.2f}"
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC per class, macro, and micro")
plt.legend(loc="lower right")
plt.savefig(config['results_and_checkpoints_folder'] + '/roc_curves.png')
# ======================================================
