def center_point(det, im0, im, arduino):
    """
    Extract the center coordinates of the first detected object of class 2
    and send the coordinates to the Arduino
    """
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Iterate through detections to find the first object of class 2
        for *xyxy, conf, cls in det:
            if int(cls) == 2:  # Check if the class is 2
                x1, y1, x2, y2 = map(int, xyxy)  # Extract coordinates

                # Calculate center coordinates
                center_x = int((x1 + x2) / 2)
                center_x -= 320
                center_y = int((y1 + y2) / 2)
                center_y -= 240
                center_y *= 2

                # Calculate center coordinates
                center_x = int((x1 + x2) / 2)
                center_x -= 320
                center_y = int((y1 + y2) / 2)
                center_y -= 240
                center_y *= 2

                # converting to 8 bit bytes
                high_xbyte = (center_x >> 8) & 0xFF  # Extract the higher byte
                low_xbyte = center_x & 0xFF          # Extract the lower byte
                high_ybyte = (center_y >> 8) & 0xFF  # Extract the higher byte
                low_ybyte = center_y & 0xFF          # Extract the lower byte
                angle = 90                           #needs to be calculated in targeting

                # Print the center coordinates
                print(f"Center coordinates: ({center_x:.0f}, {center_y:.0f})")
                ser_data = [255, high_xbyte, low_xbyte,high_ybyte, low_ybyte, angle]
                data = bytes(ser_data)
                arduino.write(data) #sending the picking cordinates to Arduino
                print(ser_data)

                break
            # if arduino.in_waiting > 0:  # print data from Arduino
            #     print(arduino.readline())