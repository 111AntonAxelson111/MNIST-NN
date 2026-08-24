





import os
import pickle
import numpy as np

from extract_mnist_data import extract_training_data_from_csv,extract_test_data_from_csv



OUT_DIR = "out"


# Find model paths
model_paths = []
for root, dirs, files in os.walk(OUT_DIR):
    for file in files:
        if file.endswith(".pkl") and "model" in file.lower():
            path = os.path.abspath(os.path.join(root, file))
            model_paths.append(path)


# Check if models exist
if len(model_paths) == 0:
    print("No models found.")
    exit()



print()
print("========================================")
print("AVAILABLE MODELS")
print("========================================")
for i, path in enumerate(model_paths):
    print()
    print(f"[{i}]")
    print(path)
print()
print("========================================")


# User selects model
while True:
    try:
        selected_index = int(input("Select model: ").strip())
        if 0 <= selected_index < len(model_paths):
            break
        print("Invalid model number.")
        
    except ValueError:
        print("Please enter a number.")


selected_path = model_paths[selected_index]


# Deserialize model
print()
print("Loading model...")
print(selected_path)

with open(selected_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
timestamp = model_data["timestamp"]
saved_accuracy = model_data["accuracy"]

print()
print("Model loaded.")
print(f"Timestamp: {timestamp}")
print(f"Saved accuracy: {saved_accuracy}")


# Load data
test_data = extract_test_data_from_csv()


while True:

    print()
    print("========================================")
    print("MODEL MENU")
    print("========================================")
    print("1 - Test accuracy")
    print("2 - Display model predictions")
    print("3 - loop exec(input('>') ")
    print("========================================")

    # get user input 
    while True:
        
        choice = input("Select: ").strip()
        
        if choice in ["1","2","3"]:
            break
        else:
            print("Please enter 1 or 2.")
    
    
    # Test accuracy
    if choice == "1":
        print()
        print("Testing model...")
        print()
        
        n_correct = 0
        n_total = 0
        
        for label, flat_img in test_data:
            
            pred_y = model(flat_img)
            
            if pred_y == label:
                n_correct+= 1
            
            n_total+= 1
        
        accuracy = n_correct / n_total
        
        print()
        print("========================================")
        print("TEST RESULT")
        print("========================================")
        print(f"Correct:   {n_correct}")
        print(f"Incorrect: {n_total - n_correct}")
        print(f"Total:     {n_total}")
        print(f"Accuracy:  {accuracy:.6f}")
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print("========================================")
    
    elif choice == "2":
        
        print()
        print("Displaying predictions.")
        print("Press ENTER to show the next image.")
        print("Press Ctrl+C to stop.")
        print()
        
        for label, flat_img in test_data:
        
            pred_y = model(flat_img)
            
            image = flat_img.reshape(28, 28)
            print()
            print()
            print()
            print()
            
            
            # Display image as terminal characters
            
            for row in image:
                line = ""
                for pxl in row:
                    
                    if 1 >= pxl > 0.8:
                        char = "██" 
                    elif 0.8 >= pxl > 0.6:
                        char = "▓▓" 
                    elif 0.6 >= pxl > 0.4:
                        char = "▒▒" 
                    elif 0.4 >= pxl > 0.2:
                        char = "▒░" 
                    elif 0.2 >= pxl > 0.1:
                        char = "░░" 
                    else:
                        char = "  " 
                    
                    line += char
                
                print(line)
                
            print("========================================")
            print(f"                                       ")
            print(f" --> {pred_y}                          ")
            print(f"                                       ")
            print("========================================")
            print()
            input("ENTER for next image...")
    
    elif choice==3:
        
        while True:
            try:
                inp= input(">")
                exec(inp)
            except Exception as e:
                print("e:")
                print(e)
