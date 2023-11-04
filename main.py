try:
    while True:
        ret, frame = drone.camera.read()  # Read a frame from the drone's camera
        sending_array = drone.detect(frame)  # Call the detect method to find objects in the frame
        sending_data = [sending_array[0], sending_array[1], sending_array[2], sending_array[3]]  # Prepare data for sending

        # Sending data to whatever system or method is set up to receive it.
        drone.sending_data(sending_data)

        # Resize the original frame for display.
        new_width = int(frame.shape[1] * 1.3275)
        new_height = frame.shape[0]
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Adjust bounding box coordinates to match the resized frame.
        x, y, w, h, label_idx = sending_array
        x_scaled = int(x * 1.3275)
        y_scaled = y  # Y doesn't change because height is the same.
        w_scaled = int(w * 1.3275)
        h_scaled = h  # Height doesn't change.

        # Draw the bounding box on the resized frame.
        if w_scaled > 0 and h_scaled > 0:
            cv2.rectangle(resized_frame, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), (0, 255, 0), 2)
            if label_idx >= 0:
                label_text = str(label_idx)
                # The placement of the text should be relative to the resized frame now.
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_w, text_h = text_size[0]
                text_x, text_y = resized_frame.shape[1] - text_w - 10, resized_frame.shape[0] - 10
                cv2.rectangle(resized_frame, (text_x, text_y + 5), (text_x + text_w, text_y - text_h - 5), (0, 255, 0), cv2.FILLED)
                cv2.putText(resized_frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Resized Frame', resized_frame)  # Display the resized frame

        # Save the resized frame as an image file.
        image_name = f"resized_{image_counter}.jpg"
        cv2.imwrite(image_name, resized_frame)
        image_counter += 1
