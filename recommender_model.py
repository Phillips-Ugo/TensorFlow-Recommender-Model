import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import ast # Import ast for literal_eval

# --- 1. Data Preparation ---

# Load your student data from the CSV file
# IMPORTANT: Ensure 'student_data.csv' is in the same directory or provide the full path.
# Also, verify that column names in your CSV match the ones used below.
df = pd.read_csv('student_data.csv')

# Convert string lists to Python lists safely
def safe_convert_to_list(x):
    if isinstance(x, str):
        try:
            # Use ast.literal_eval for safe evaluation of string literals
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # Return empty list if parsing fails
            return []
    return x

# Apply the conversion to relevant columns
# These columns are expected to contain string representations of lists (e.g., "['item1', 'item2']")
df['interests'] = df['interests'].apply(safe_convert_to_list)
df['completed_courses'] = df['completed_courses'].apply(safe_convert_to_list)

# IMPORTANT FIX: Fill any NaN values in categorical columns with a placeholder string
# This ensures that StringLookup vocabularies are clean and don't contain 'nan' strings.
df['major'].fillna('UNKNOWN_MAJOR', inplace=True)
df['career_goal'].fillna('UNKNOWN_GOAL', inplace=True)
df['best_fit_recommended_class'].fillna('UNKNOWN_CLASS', inplace=True)


# Get unique values for categorical features and classes
# Ensure unique values are explicitly strings
unique_majors = df['major'].unique().astype(str)
unique_career_goals = df['career_goal'].unique().astype(str)
unique_classes = df['best_fit_recommended_class'].unique().astype(str) # All possible classes

# Multi-hot encoding for interests and completed courses
# Flatten all lists of interests and courses to get unique values
# Handle cases where 'interests' or 'completed_courses' might be empty lists
all_interests = np.unique(np.concatenate(df['interests'].values if len(df['interests']) > 0 else [[]]))
all_courses = np.unique(np.concatenate(df['completed_courses'].values if len(df['completed_courses']) > 0 else [[]]))

# Create mappings from item to index
interest_to_idx = {v: i for i, v in enumerate(all_interests)}
course_to_idx = {v: i for i, v in enumerate(all_courses)}

# Function to encode a list of items into a multi-hot vector
def encode_multi_hot(items, mapping):
    encoding = np.zeros(len(mapping), dtype=np.float32)
    for item in items:
        if item in mapping:
            encoding[mapping[item]] = 1.0
    return encoding

# Apply multi-hot encoding to DataFrame columns
df['interests_encoded'] = df['interests'].apply(lambda x: encode_multi_hot(x, interest_to_idx))
df['courses_encoded'] = df['completed_courses'].apply(lambda x: encode_multi_hot(x, course_to_idx))

# --- 2. Model Definitions ---

class StudentModel(tf.keras.Model):
    def __init__(self, unique_majors, unique_career_goals, num_interests, num_courses):
        super().__init__()
        # Embedding layer for 'major'
        self.major_embedding = tf.keras.Sequential([
            # Explicitly convert vocabulary to tf.string tensor
            layers.StringLookup(vocabulary=tf.constant(unique_majors, dtype=tf.string), mask_token=None),
            layers.Embedding(input_dim=len(unique_majors) + 1, output_dim=32)
        ])
        # Embedding layer for 'career_goal'
        self.career_goal_embedding = tf.keras.Sequential([
            # Explicitly convert vocabulary to tf.string tensor
            layers.StringLookup(vocabulary=tf.constant(unique_career_goals, dtype=tf.string), mask_token=None),
            layers.Embedding(input_dim=len(unique_career_goals) + 1, output_dim=32)
        ])
        # Dense layers for multi-hot encoded interests and completed courses
        self.interests_dense = layers.Dense(32, activation='relu')
        self.courses_dense = layers.Dense(32, activation='relu')

        # Normalization layer for 'gpa'
        self.gpa_normalization = layers.Normalization(axis=None)

        # Combined dense layers for the student embedding
        self.dense_layers = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32)
        ])

    def call(self, inputs):
        # Process each input feature
        major_emb = self.major_embedding(inputs["major"])
        career_emb = self.career_goal_embedding(inputs["career_goal"])

        # Ensure multi-hot inputs are float32
        interest_emb = self.interests_dense(tf.cast(inputs["interests"], tf.float32))
        course_emb = self.courses_dense(tf.cast(inputs["completed_courses"], tf.float32))

        # Normalize GPA and reshape for concatenation
        gpa = self.gpa_normalization(tf.reshape(inputs["gpa"], (-1, 1)))

        # Concatenate all processed features
        concatenated = tf.concat([major_emb, career_emb, interest_emb, course_emb, gpa], axis=1)

        # Pass through final dense layers to get the student embedding
        return self.dense_layers(concatenated)

class CourseModel(tf.keras.Model):
    def __init__(self, unique_classes):
        super().__init__()
        # StringLookup to convert class names to integer IDs
        # Explicitly convert vocabulary to tf.string tensor
        self.class_lookup = layers.StringLookup(vocabulary=tf.constant(unique_classes, dtype=tf.string), mask_token=None)
        # Embedding layer for class IDs
        self.class_embedding = layers.Embedding(input_dim=len(unique_classes) + 1, output_dim=32)

    def call(self, class_names):
        # Look up class IDs and then get their embeddings
        class_ids = self.class_lookup(class_names)
        return self.class_embedding(class_ids)

class RecommenderModel(tfrs.Model):
    def __init__(self, student_model, course_model, unique_classes):
        super().__init__()
        self.student_model = student_model
        self.course_model = course_model

        # The FactorizedTopK metric needs a dataset of candidates to compute metrics.
        # It expects (embedding, id) pairs.
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                # Create a dataset of unique classes for candidates, mapping each to (embedding, class_name)
                candidates=tf.data.Dataset.from_tensor_slices(unique_classes).batch(128).map(
                    lambda class_name: (self.course_model(class_name), class_name)
                ),
                ks=[5, 10, 20] # Top-K accuracy metrics
            )
        )

    def compute_loss(self, inputs, training=False):
        # inputs is a tuple (features, labels) from the dataset
        features, labels = inputs

        # Get student embeddings from the student model
        student_embeddings = self.student_model({
            "major": features["major"],
            "gpa": features["gpa"],
            "career_goal": features["career_goal"],
            "interests": features["interests"],
            "completed_courses": features["completed_courses"]
        })

        # Get course embeddings for the recommended class (target)
        course_embeddings = self.course_model(labels) # Use 'labels' directly as it's the target class name

        # Compute the retrieval loss
        return self.task(student_embeddings, course_embeddings)

# --- 3. Dataset Preparation ---

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    # Pop the target column to use as labels
    labels = dataframe.pop('best_fit_recommended_class')

    # Create a dictionary of features, converting relevant columns to tensors
    features = {
        # Explicitly set dtype=tf.string for string features
        'major': tf.convert_to_tensor(dataframe['major'], dtype=tf.string),
        'gpa': tf.convert_to_tensor(dataframe['gpa'], dtype=tf.float32),
        'career_goal': tf.convert_to_tensor(dataframe['career_goal'], dtype=tf.string),
        # Stack the encoded arrays into a single tensor
        'interests': tf.convert_to_tensor(np.stack(dataframe['interests_encoded'].values), dtype=tf.float32),
        'completed_courses': tf.convert_to_tensor(np.stack(dataframe['courses_encoded'].values), dtype=tf.float32),
    }

    # Create a TensorFlow Dataset from the features and labels
    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Add prefetching for performance

# --- 4. Training Setup ---

# Split data into training and validation sets
train_size = int(0.8 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]

# Initialize models with necessary vocabulary sizes
student_model = StudentModel(unique_majors, unique_career_goals, len(all_interests), len(all_courses))
course_model = CourseModel(unique_classes)

# Adapt GPA normalization layer using the GPA data from the training set
# It's crucial to adapt before creating the tf.data.Dataset for training,
# or adapt using the raw data from the DataFrame.
student_model.gpa_normalization.adapt(train_df['gpa'].values)

# Create TensorFlow Datasets
train_ds = df_to_dataset(train_df)
val_ds = df_to_dataset(val_df, shuffle=False)

# Create and compile the recommender model
model = RecommenderModel(student_model, course_model, unique_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001))

# Train the model
print("Starting model training...")
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
print("Model training complete.")

# --- 5. Making Predictions (Example) ---

# Create a retrieval model for serving predictions
# This model wraps the student model and uses a BruteForce layer for efficient candidate retrieval
# You could also use a more sophisticated index like ScaNN for larger datasets
index = tfrs.layers.factorized_top_k.BruteForce(student_model.dense_layers) # Use the output layer of student_model
index.index_from_dataset(
    # Ensure the dataset for indexing also provides (embedding, id) pairs
    tf.data.Dataset.from_tensor_slices(unique_classes).batch(128).map(lambda class_name: (course_model(class_name), class_name))
)

# Example student data for prediction
# IMPORTANT: Ensure the example data matches the structure and potential 'UNKNOWN' placeholders
# if you want to test with values that might have been missing in the original dataset.
test_student_data = {
    "major": tf.constant(["Computer Science"]),
    "gpa": tf.constant([3.7], dtype=tf.float32),
    "career_goal": tf.constant(["Software Engineer"]),
    "interests": tf.constant([encode_multi_hot(['AI', 'Web Development'], interest_to_idx)], dtype=tf.float32),
    "completed_courses": tf.constant([encode_multi_hot(['CS101', 'MATH201'], course_to_idx)], dtype=tf.float32),
}

# Get top-K recommendations for the test student
print("\nGenerating recommendations for a test student:")
_, top_classes = index(test_student_data)
print(f"Top recommended classes: {top_classes[0].numpy().tolist()}")

# You can also get the scores
# scores, classes = index(test_student_data)
# for i, (score, class_name) in enumerate(zip(scores[0], classes[0])):
#     print(f"Rank {i+1}: {class_name.decode('utf-8')} (Score: {score:.4f})")

# Save the student model
student_model.save("saved_models/student_model")

# Save the course model
course_model.save("saved_models/course_model")
