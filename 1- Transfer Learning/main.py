import os
import numpy
import keras
import random
import zipfile
import tempfile
import requests
import tensorflow
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import History
from numpy import ndarray, dtype
from typing import Any
from PIL.ImageFile import ImageFile
from PIL import UnidentifiedImageError


def set_permissions_recursive(root_directory: str, permission_mode: int = 0o777) -> None:
    """
    Recursively sets the permissions of the specified directory, and all its subdirectories and files,
    to the given permission mode.
    :param root_directory: The path of the directory whose permissions should be changed.
    :param permission_mode: The permission mode to set (default is 0o777).
    """
    os.chmod(root_directory, permission_mode)

    # os.walk returns a tuple: (current_directory, list_of_subdirectories, list_of_files)
    for current_directory, subdirectories, file_names in os.walk(root_directory):
        for subdirectory in subdirectories:
            subdirectory_path: str = os.path.join(current_directory, subdirectory)
            os.chmod(subdirectory_path, permission_mode)
        for file_name in file_names:
            file_path: str = os.path.join(current_directory, file_name)
            os.chmod(file_path, permission_mode)


def download_dataset(zip_url: str, destination_path: str) -> str:
    """
    Downloads a zip file and extracts it to the specified directory.
    :param zip_url: The URL to the zip file.
    :param destination_path: The directory the contents of the zip file should be extracted to.
    :return: The absolute path to the directory with the contents of the zip file.
    """
    absolute_destination_path: str = os.path.abspath(destination_path)
    print('> Downloading dataset')
    print(f'>> from: {zip_url}')
    print(f'>> to: {absolute_destination_path}')

    # Create the destination directory if it doesn't exist
    os.makedirs(absolute_destination_path, exist_ok=True)

    # Download the zip file to a temporary file in the system's temp directory
    with requests.get(zip_url, stream=True) as response:
        response.raise_for_status()  # Check for HTTP request errors

        # Create a named temporary file with .zip suffix
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            zip_file_path: str = temp_file.name

            # Download the zip file in chunks
            for data_chunk in response.iter_content(chunk_size=8192):
                if data_chunk is not None:
                    temp_file.write(data_chunk)

    # Extract the zip file to the destination directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(absolute_destination_path)

    # Delete the temporary zip file
    os.remove(zip_file_path)

    set_permissions_recursive(absolute_destination_path)

    return absolute_destination_path


def is_directory_empty(directory_path: str) -> bool:
    """
    Checks if the specified directory is empty.
    :param directory_path: The absolute path to the directory.
    :return: True if the directory is empty or does not exist, False otherwise.
    """
    return not os.path.exists(directory_path) or len(os.listdir(directory_path)) == 0


def process_image(image_path: str, target_height: int, target_width: int) -> tuple[ImageFile | None, ndarray[tuple, dtype]]:
    """
    Loads and preprocesses an image.
    :param image_path: The absolute path to the image.
    :param target_height: The desired height for the preprocessed image.
    :param target_width: The desired width for the preprocessed image.
    :return: The image and the input vector.
    """
    try:
        image: ImageFile = keras.preprocessing.image.load_img(image_path, target_size=(target_height, target_width))
        input_vector: ndarray[tuple, dtype] = keras.preprocessing.image.img_to_array(image)
        input_vector = numpy.expand_dims(input_vector, axis=0)
        input_vector = keras.applications.imagenet_utils.preprocess_input(input_vector)
        return image, input_vector
    except UnidentifiedImageError:
        print(f'Failed to process {image_path}')
        return None, numpy.empty((0,), dtype=object)


def build_neural_network(training_data: ndarray[Any, dtype[Any]], image_class_amount: int) -> Model:
    """
    Creates a neural network from scratch.
    :param training_data: The training data.
    :param image_class_amount: The amount of image classes present in the training data.
    :return: The neural nerwork.
    """
    model: Sequential = Sequential()

    print("Input dimensions: ", training_data.shape[1:])

    model.add(Conv2D(32, (3, 3), input_shape=training_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(image_class_amount))
    model.add(Activation('softmax'))

    # Use categorical cross-entropy loss function and adadelta optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def create_model_for_finetuning(image_class_amount: int) -> Model:
    """
    Creates a model for fine-tuning.
    :param image_class_amount: The amount of image classes present in the training data.
    :return: The neural network model.
    """
    vgg: Model = keras.applications.VGG16(weights='imagenet', include_top=True)
    vgg.summary()

    # VGG's input layer
    input_layer: Input = vgg.input

    # Make a new softmax layer with len(image_classes) neurons
    new_classification_layer: Dense = Dense(image_class_amount, activation='softmax')

    # Connect our new layer to the second to last layer in VGG, and grab a reference to it
    output_layer: Dense = new_classification_layer(vgg.layers[-2].output)

    # Create a new network between input and output
    finetuned_model: Model = Model(input_layer, output_layer)

    # Freeze all weights, making all layers untrainable (except for the last layer)
    for layer in finetuned_model.layers[:-1]:
        layer.trainable = False

    # Ensure the last layer is trainable (not frozen)
    finetuned_model.layers[-1].trainable = True

    finetuned_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    finetuned_model.summary()

    return finetuned_model


def predict(model: Model, image_path: str) -> ndarray[tuple, dtype]:
    """
    Makes a prediction with the specified neural network model.
    :param model: The neural network model.
    :param image_path: The absolute or relative path the image is located at.
    :return: The model's prediction.
    """
    vectors: ndarray[tuple, dtype] = process_image(image_path, 224, 224)[1]
    return model.predict(vectors)



def main() -> None:
    zip_url: str = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
    destination_path: str = os.path.abspath('data')
    dataset_path: str = os.path.join(download_dataset(zip_url, destination_path) if is_directory_empty(destination_path) else destination_path, 'PetImages')
    image_classes: list[str] = os.listdir(dataset_path)

    # Load and pre-process all images in memory
    samples: list[dict[str, ndarray[tuple, dtype] | int | str]] = []

    for index, image_class in enumerate(image_classes):
        for image_name in os.listdir(os.path.join(dataset_path, image_class))[
        :2000]:  # Limit to 2000 images per class due to memory limitations
            image_path: str = os.path.join(dataset_path, image_class, image_name)
            vector = process_image(image_path, 224, 224)[1]
            if (vector.any()):
                samples.append({ 'x': numpy.array(vector[0]), 'y': index, 'img_path': image_path })

    print(f'Loaded {len(samples)} images.')

    # Shuffle the sample data
    random.shuffle(samples)

    # Create training / validation / test splits (70%, 15%, 15%)
    train_split: float = 0.7
    val_split: float = 0.15

    validation_index: int = int(train_split * len(samples))
    test_index: int = int((train_split + val_split) * len(samples))

    training_set: list[dict[str, ndarray[tuple, dtype] | int]] = samples[:validation_index]
    validation_set: list[dict[str, ndarray[tuple, dtype] | int]] = samples[validation_index:test_index]
    testing_set: list[dict[str, ndarray[tuple, dtype] | int]] = samples[test_index:]

    # Separate data for labels.
    x_train: ndarray[Any, dtype[Any]] = numpy.array([sample["x"] for sample in training_set])
    y_train: list[int] = [sample["y"] for sample in training_set]
    x_val: ndarray[Any, dtype[Any]] = numpy.array([sample["x"] for sample in validation_set])
    y_val: list[int] = [sample["y"] for sample in validation_set]
    x_test: ndarray[Any, dtype[Any]] = numpy.array([sample["x"] for sample in testing_set])
    y_test: list[int] = [sample["y"] for sample in testing_set]

    # Normalize the data
    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Convert labels to one-hot vectors
    y_train: ndarray[Any, dtype[Any]] = keras.utils.to_categorical(y_train, len(image_classes))
    y_val: ndarray[Any, dtype[Any]] = keras.utils.to_categorical(y_val, len(image_classes))
    y_test: ndarray[Any, dtype[Any]] = keras.utils.to_categorical(y_test, len(image_classes))

    # Print summary
    print(f'Finished loading {len(samples)} images from {len(image_classes)} categories')
    print(f'Training: {len(x_train)} | Validation: {len(x_val)} | Testing: {len(x_test)}')
    print(f"Training data shape: ", x_train.shape)
    print(f"Training labels shape: ", y_train.shape)

    # Build a neural network from scratch
    model: Model = build_neural_network(x_train, len(image_classes))

    print('Starting training of the neural network created from scratch.')
    model.fit(x_train, y_train, batch_size=128, epochs=18, validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # Download the VGG16 model with the 'imagenet' training weights.
    print('Downloading the VGG16 pre-trained model for fine-tuning.')
    finetuned_model: Model = create_model_for_finetuning(len(image_classes))

    print('Starting fine-tuning...')
    finetuned_model.fit(x_train, y_train, batch_size=128, epochs=18, validation_data=(x_val, y_val))

    loss, accuracy = finetuned_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # Making predictions
    cat_path: str = 'data/PetImages/Cat/0.jpg'
    dog_path: str = 'data/PetImages/Dog/0.jpg'

    print(f'Cat prediction (old model)       : {predict(model, cat_path)}')
    print(f'Cat prediction (fine-tuned model): {predict(finetuned_model, cat_path)}')

    print(f'Dog prediction (old model)       : {predict(model, dog_path)}')
    print(f'Dog prediction (fine-tuned model): {predict(finetuned_model, dog_path)}')


if __name__ == '__main__':
    main()
