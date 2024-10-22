import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Capture image using Streamlit
image = st.camera_input("Capture Barcode Image")

if image is not None:
    # Open the image from Streamlit as a PIL Image
    img_pil = Image.open(image)

    # Convert the PIL Image to a NumPy array
    img = np.array(img_pil)

    # Convert RGB to BGR format for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Initialize OpenCV Barcode Detector
    bd = cv2.barcode.BarcodeDetector()

    # Detect and decode barcodes
    decoded_info, decoded_type, points = bd.detectAndDecode(img_bgr)

    # Check if barcodes were detected
    if decoded_info:  # Check if there is any decoded info
        st.success("Barcode Detected!")
        st.write(f"Decoded Data: {decoded_info}")  # This is the actual barcode data
        st.write(f"Decoded Type: {decoded_type}")

        if points is not None:  # Check if points are available
            # Draw bounding box and decoded info
            img_bgr = cv2.polylines(img_bgr, points.astype(int), True, (0, 255, 0), 3)

            for s, p in zip(decoded_info, points):
                img_bgr = cv2.putText(img_bgr, s, p[1].astype(int),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Convert the processed image back to RGB format and display it in Streamlit
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Processed Image with Barcode", use_column_width=True)
        else:
            st.warning("Points not available to draw bounding box.")
    else:
        st.error("No barcode detected.")





######### BARCODE WORKING BUT WITH TEMPORARILY SAVING IMAGE ###########

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile

# # Capture image using Streamlit
# image = st.camera_input("Capture Barcode Image")

# if image is not None:
#     # Open the image from Streamlit as a PIL Image
#     img_pil = Image.open(image)

#     # Save the image to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
#         img_pil.save(temp.name)
#         temp_img_path = temp.name

#     # Now, read the image from the temporary file using OpenCV
#     img = cv2.imread(temp_img_path)

#     # Initialize OpenCV Barcode Detector
#     bd = cv2.barcode.BarcodeDetector()

#     # Detect and decode barcodes
#     decoded_info, decoded_type, points = bd.detectAndDecode(img)

#     # Check if barcodes were detected
#     if decoded_info:  # Check if there is any decoded info
#         st.success("Barcode Detected!")
#         st.write(f"Decoded Data: {decoded_info}")  # This is the actual barcode data
#         st.write(f"Decoded Type: {decoded_type}")

#         if points is not None:  # Check if points are available
#             # Draw bounding box and decoded info
#             img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 3)

#             for s, p in zip(decoded_info, points):
#                 img = cv2.putText(img, s, p[1].astype(int),
#                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

#             # Convert the processed image back to RGB format and display it in Streamlit
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             st.image(img_rgb, caption="Processed Image with Barcode", use_column_width=True)
#         else:
#             st.warning("Points not available to draw bounding box.")
#     else:
#         st.error("No barcode detected.")








######## QRCODE










# import cv2
# import numpy as np
# import streamlit as st

# # Capture the image from the camera input
# image = st.camera_input("Show QR code")

# if image is not None:
#     # Convert image bytes to OpenCV format
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Check the shape of the captured image
#     st.write(f"Captured image shape: {cv2_img.shape}")

#     # Preprocess the image: convert to grayscale
#     gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

#     # Optional: Show grayscale image for debugging
#     st.image(gray_img, caption="Grayscale Image", channels="GRAY")

#     # Initialize the QR code detector
#     detector = cv2.QRCodeDetector()

#     # Detect and decode the QR code
#     data, bbox, straight_qrcode = detector.detectAndDecode(gray_img)

#     # If QR code is detected, show bounding box and data
#     if bbox is not None and data:
#         st.success("QR Code Detected!")
#         st.write(f"Decoded Data: {data}")

#         # Draw bounding box on the original image
#         for i in range(len(bbox)):
#             cv2.line(cv2_img, tuple(bbox[i][0]), tuple(bbox[(i + 1) % len(bbox)][0]), color=(0, 255, 0), thickness=2)

#         # Display the image with bounding box
#         st.image(cv2_img, channels="BGR")
#     else:
#         st.warning("No QR code detected. Please try again!")

#         # Debugging: Show the image without QR code for analysis
#         st.image(cv2_img, caption="Captured Image", channels="BGR")

#         # Print bbox and data for debugging
#         st.write(f"Bounding box: {bbox}")
#         st.write(f"Decoded Data: {data}")





# import cv2
# import numpy as np
# import streamlit as st

# # Capture the image from the camera input
# image = st.camera_input("Show QR code")

# if image is not None:
#     # Convert image bytes to OpenCV format
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Preprocess the image: convert to grayscale
#     gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

#     # Initialize the QR code detector
#     detector = cv2.QRCodeDetector()

#     # Detect and decode the QR code
#     data, bbox, straight_qrcode = detector.detectAndDecode(gray_img)

#     # If QR code is detected, show bounding box and data
#     if bbox is not None and data:
#         st.success("QR Code Detected!")
#         st.write(f"Decoded Data: {data}")

#         # Draw bounding box on the original image
#         for i in range(len(bbox)):
#             cv2.line(cv2_img, tuple(bbox[i][0]), tuple(bbox[(i + 1) % len(bbox)][0]), color=(0, 255, 0), thickness=2)

#         # Display the image with bounding box
#         st.image(cv2_img, channels="BGR")
#     else:
#         st.warning("No QR code detected. Please try again!")

#         # Debugging: Show the image without QR code for analysis
#         st.image(cv2_img, caption="Captured image", channels="BGR")

#         # Optionally, print bbox and data for debugging
#         st.write(f"Bounding box: {bbox}")
#         st.write(f"Decoded Data: {data}")


# import cv2
# import numpy as np
# import streamlit as st

# # Capture the image from the camera input
# image = st.camera_input("Show QR code")

# if image is not None:
#     # Convert image bytes to OpenCV format
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Initialize the QR code detector
#     detector = cv2.QRCodeDetector()

#     # Detect and decode the QR code
#     data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

#     # If QR code is detected, show bounding box and data
#     if bbox is not None and data:
#         st.success("QR Code Detected!")
#         st.write(f"Decoded Data: {data}")

#         # Draw bounding box on the image
#         for i in range(len(bbox)):
#             cv2.line(cv2_img, tuple(bbox[i][0]), tuple(bbox[(i + 1) % len(bbox)][0]), color=(0, 255, 0), thickness=2)

#         # Display the image with bounding box
#         st.image(cv2_img, channels="BGR")
#     else:
#         st.warning("No QR code detected. Please try again!")





# import cv2
# import numpy as np
# import streamlit as st
# from camera_input_live import camera_input_live

# "# Streamlit camera input live Demo"
# "## Try holding a qr code in front of your webcam"

# if "found_qr" not in st.session_state:
#     st.session_state.found_qr = False

# if "qr_code_image" not in st.session_state:
#     st.session_state.qr_code_image = None

# if not st.session_state["found_qr"]:
#     image = camera_input_live()
# else:
#     image = st.session_state.qr_code_image

# if image is not None:
#     st.image(image)
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     detector = cv2.QRCodeDetector()

#     data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

#     if data:
#         st.session_state["found_qr"] = True
#         st.session_state["qr_code_image"] = image
#         st.write("# Found QR code")
#         st.write(data)
#         with st.expander("Show details"):
#             st.write("BBox:", bbox)
#             st.write("Straight QR code:", straight_qrcode)






# import cv2
# import numpy as np
# import streamlit as st

# image = st.camera_input("Show QR code")

# if image is not None:
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     detector = cv2.QRCodeDetector()

#     data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

#     st.write("Here!")
#     st.write(data)




# import numpy as np
# import streamlit as st

# from camera_input_live import camera_input_live

# "# Streamlit camera input live Demo"
# "## Try holding a qr code in front of your webcam"

# image = camera_input_live()

# if image is not None:
#     st.image(image)
#     bytes_data = image.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     detector = cv2.QRCodeDetector()

#     data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

#     if data:
#         st.write("# Found QR code")
#         st.write(data)
#         with st.expander("Show details"):
#             st.write("BBox:", bbox)
#             st.write("Straight QR code:", straight_qrcode)
