from PIL import Image, ImageOps
import os

def menu(img):
    print("\nChoose an image manipulation:")
    print("1. Invert colors")
    print("2. Convert to grayscale")
    print("3. Increase brightness")
    print("4. Exit the program")
    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        img = ImageOps.invert(img.convert("RGB"))
    elif choice == "2":
        img = ImageOps.grayscale(img)
    elif choice == "3":
        img = ImageOps.autocontrast(img)
    elif choice == "4":
        print("Exiting the Program")
        return None
    else:
        print("Invalid choice. No manipulation applied.")
        return img

    return img

def Manipulation(img):
    while True:
        manipulated_img = menu(img)

        if manipulated_img is None:
                print("Exiting")
                break
            
        save= input("Enter a path to save the manipulated image ")
        if not save.lower().endswith(('.png', '.jpg', '.jpeg')):
                print("No valid extension detected. Adding .jpg as the default extension.")
                save+= ".jpg"
        try:
                manipulated_img.save(save)
                print(f"Manipulated image saved to {save}")
        except:
                print(f"Error saving image")

        img = manipulated_img


def main():
    file_path = input("Enter the file path to the image: ")
    
    if not os.path.exists(file_path):
        print("File path does not exist.")
        return
    try:
        print(f"Loading image from {file_path}...")
        img = Image.open(file_path)
        print("Image loaded successfully.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    Manipulation(img)



if __name__ == "__main__":
    main()
