validation_model = tf.keras.models.load_model('complete_model (1).keras')
def predict_image(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    if img.size[0] < 128 or img.size[1] < 128:  # Check original size before resizing
        return None, "low_resolution", "Image resolution too low (minimum 128x128 pixels)"
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    is_xray = validation_model.predict(img_array, verbose=0)[0][0] > 0.5  # Threshold
    if not is_xray:
        return None, "invalid_xray", "Image does not appear to be a chest X-ray"
    pred = model.predict(img_array, verbose=0)
    return pred, None, None