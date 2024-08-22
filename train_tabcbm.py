import scipy
import joblib
import sklearn
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

import tensorflow as tf

from tabcbm.models.architectures import construct_encoder, construct_decoder
from tabcbm.models.architectures import construct_end_to_end_model
from tabcbm.models.tabcbm import TabCBM


data_dir = 'D:\\PycharmProjects\\AMMISproject\\data\\processed_data'
dataset = 'dataco'

x_train_std = joblib.load(osp.join(data_dir, dataset, 'x_train_std.joblib'))
x_test_std = joblib.load(osp.join(data_dir, dataset, 'x_test_std.joblib'))

x_train = joblib.load(osp.join(data_dir, dataset, 'x_train.joblib'))
x_test = joblib.load(osp.join(data_dir, dataset, 'x_test.joblib'))

y_train = joblib.load(osp.join(data_dir, dataset, 'y_train.joblib'))
y_test = joblib.load(osp.join(data_dir, dataset, 'y_test.joblib'))

print('Shape of the training set: ', x_train_std.shape)
print('Shape of the test set: ', x_test_std.shape)
print('Shape of the trainigb targets: ', y_train.shape)

# Parameters defining the architecture we will use

input_shape = x_train_std.shape[1:]
num_outputs = len(set(y_train))
encoder_units = [16, 16]
decoder_units = [16]
latent_dims = 16
learning_rate = 0.001
validation_size = 0.1

print('Input shape: ', input_shape)
print('Number of outputs: ', num_outputs)

# Next, we build the feature to latent code encoder model (i.e., phi)
encoder = construct_encoder(input_shape, encoder_units, latent_dims)
encoder.summary()

# Then, we build the concept to label model  (i.e., the label predictor f)

decoder_inputs = tf.keras.Input(shape=[latent_dims])
decoder_graph = construct_decoder(decoder_units, num_outputs)
decoder = tf.keras.Model(
    decoder_inputs,
    decoder_graph(decoder_inputs),
    name="decoder",
)
decoder.summary()

end_to_end_model, encoder, decoder = construct_end_to_end_model(input_shape,
                                                                encoder,
                                                                decoder,
                                                                num_outputs,
                                                                learning_rate)

end_to_end_model.summary()

pretrain_epochs = 30
batch_size = 512
pretrain_hist = end_to_end_model.fit(
    x=x_train_std,
    y=y_train,
    epochs=pretrain_epochs,
    batch_size=batch_size,
    validation_split=validation_size,
    verbose=1,
)

# We will accumulate all metrics/results in the same dictionary
results = {}

# Make test predictions for the test set
end_to_end_preds = end_to_end_model.predict(
    x_test_std,
    batch_size=batch_size,
)

# Get accuracy/AUC using the corresponding test labels
# Dealing with simple binary outputs
if np.min(end_to_end_preds) < 0.0 or np.max(end_to_end_preds) > 1:
    # Then we assume that we have outputed logits
    end_to_end_preds = tf.math.sigmoid(end_to_end_preds).numpy()
end_to_end_preds = (end_to_end_preds >= 0.5).astype(np.int32)
results['pre_train_acc'] = sklearn.metrics.accuracy_score(
    y_test,
    end_to_end_preds,
)
results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
    y_test,
    end_to_end_preds,
)
print(f"Pretrained model task accuracy: {results['pre_train_acc']*100:.2f}%")


# Construct TabCBM

# Construct the training set's empirical covariance matrix
# NOTE: This step can be very computationally expensive/intractable in large
#       datasets. In those cases, one may ignore the covariance matrix when
#       performing TabCBM's pretraining at the potential cost of performance or
#       more accurate concept discovery.
cov_mat = np.corrcoef(x_train_std.T)
print(cov_mat)

# Number of concepts we want to discover
n_concepts = 4

# Set the weights for the different regularisers in the loss
coherence_reg_weight = 0.1  # $lambda_{co}
diversity_reg_weight = 5  # $lambda_{div}
feature_selection_reg_weight = 5  # $lambda_{spec}
gate_estimator_weight = 10  # Gate prediction regularizer for SEFS's pre-text task

# Select how many neighbors to use for the coherency loss (must be less than the batch size!)
top_k = 256

# Generate a dictionary with the parameters to use for TabCBM as we will have
# to use the same parameters twice:
tab_cbm_params = dict(
    features_to_concepts_model=encoder,  # The $\phi$ sub-model
    concepts_to_labels_model=decoder,  # The $f$ sub-model
    latent_dims=latent_dims,  # The dimensionality of the concept embeddings $m$
    n_concepts=n_concepts,  # The number of concepts to discover $k^\prime$
    cov_mat=cov_mat,  # The empirical covariance matrix
    loss_fn=end_to_end_model.loss,  # The downstream task loss function
    # Then we provide all the regularizers weights
    coherence_reg_weight=coherence_reg_weight,
    diversity_reg_weight=diversity_reg_weight,
    feature_selection_reg_weight=feature_selection_reg_weight,
    gate_estimator_weight=gate_estimator_weight,
    top_k=top_k,

    # And indicate that we will not be providing any supervised concepts! Change
    # this is training concepts (e.g., `c_train`) are provided/known during
    # training
    n_supervised_concepts=0,
    concept_prediction_weight=0,

    # The accuracy metric to use for logging performance
    acc_metric=(
        lambda y_true, y_pred: tf.math.reduce_mean(
            tf.keras.metrics.sparse_categorical_accuracy(
                y_true,
                y_pred,
            )
        )
    ),

    # ANd architectural details of the self-supervised reconstruction modules
    concept_generator_units=[64],
    rec_model_units=[64],
)

# Mask Generator Self-supervised Training

# Next, we proceed to do the self-supervised training of the mask
# generators for TabCBM. For this, we will follow a similar approach
# to that of SEFS. Our TabCBM module allows one to do this by setting
# the self_supervised_mode flag to True before calling the .fit() method:

# We can now construct our TabCBM model which we will first self-supervise!
ss_tabcbm = TabCBM(self_supervised_mode=True,  **tab_cbm_params)
ss_tabcbm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate,))
ss_tabcbm.summary()


# And we are ready to do the SS pretraining of the mask generators for a total
# of 50 epochs
self_supervised_train_epochs = 50
print("TabCBM self-supervised training stage...")
ss_tabcbm_hist = ss_tabcbm.fit(
    x=x_train_std,
    y=y_train,
    validation_split=validation_size,
    epochs=self_supervised_train_epochs,
    batch_size=batch_size,
    verbose=1,
)
print("\tTabCBM self-supervised training completed")


# First we will instantiate a new TabCBM that is NOT in self-supervised mode,
# and we will load its weights so that they are the same as the model whose
# mask generators have been pre-trained using the SS loss.
tabcbm_supervised = TabCBM(
    self_supervised_mode=False,
    # Notice how we provide as concept generators the concept generators of the
    # SS TabCBM:
    concept_generators=ss_tabcbm.concept_generators,
    # as well as the feature probability masks:
    prior_masks=ss_tabcbm.feature_probabilities,
    **tab_cbm_params,
)
tabcbm_supervised.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
tabcbm_supervised.summary()
# Next, we perform the end-to-end training of this architecture
#####################

# Number of maximum epochs to train
max_epochs = 1500

# Time to do the end-to-end training!
tabcbm_hist = tabcbm_supervised.fit(
    x=x_train_std,
    y=y_train,
    validation_split=validation_size,
    epochs=max_epochs,
    batch_size=batch_size,
    verbose=1,
)
print("\tTabCBM supervised training completed")
