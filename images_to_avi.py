import os
import cv2


def images_to_avi(image_prefix, base_output_filename, fps=10):
    files = os.listdir()
    jpg_files = [file for file in files if file.startswith(image_prefix) and file.endswith('.jpg')]

    jpg_files.sort(key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))

    if not jpg_files:
        print("No jpg files found with the given prefix.")
        return

    img = cv2.imread(jpg_files[0])
    if img is None:
        print(f"Error reading the image: {jpg_files[0]}")
        return

    height, width, layers = img.shape

    combinations = [('XVID', 'avi')]

    for codec, ext in combinations:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_filename = f"{base_output_filename}_{codec}.{ext}"
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        for file in jpg_files:
            img = cv2.imread(file)
            if img is not None:
                out.write(img)
            else:
                print(f"Error reading the image: {file}")

        out.release()
        print(f"Saved video with {codec} codec to {output_filename}")


images_to_avi("", "output")