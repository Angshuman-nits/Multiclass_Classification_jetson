from email.mime import image
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from transform import ToImageGraph
import torchvision.transforms as T



import torch
import torch.nn as nn
import torch_geometric.nn as gnn

def browse_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = Image.open(file_path)
        image2 = image.resize((300, 300))  # Resize the image to fit the window
        photo = ImageTk.PhotoImage(image2)
        image_label.configure(image=photo)
        image_label.image = photo

        pred_label.configure(text="")

normal_transform = ToImageGraph(100,1.5)

aug_functions = [
    T.RandomHorizontalFlip(p=1), 
    T.RandomAdjustSharpness(p=1,sharpness_factor=2),
    T.RandomAutocontrast(p=1),
    T.RandomVerticalFlip(p=1),
]

aug_transforms = [T.Compose([aug_function, ToImageGraph(100,1.5)]) for aug_function in aug_functions]
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(3,32)
        self.conv2 = gnn.GCNConv(32,64)
        self.conv3 = gnn.GCNConv(64,128)
        self.global_pool = gnn.global_mean_pool
        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(32,4)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.global_pool(x, batch)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x
    
model = torch.load('GAT.pt')
model.eval()
model.to(device)

pred_map={0:'exudative',1:'healthy',2:'rhegmatogenous',3:'tractional'}
#classes = ['exudative', 'healthy', 'rhegmatogenous', 'tractional']

def perform_action():
    img=Image.open(file_path)
    data = normal_transform(img)
    data= data.to(device)
    out = model(data)
    _,pred = torch.max(out,1)
    pred_label.configure(text=f"Prediction: {pred_map[pred.item()]}")
    pass

# Create the main window
window = tk.Tk()

# Create the browse button
browse_button = tk.Button(window, text="Browse", command=browse_image)
browse_button.pack()

# Create the image label
image_label = tk.Label(window)
image_label.pack()
pred_label = tk.Label(window)
pred_label.pack()

# Create the action button
action_button = tk.Button(window, text="Action", command=perform_action)
action_button.pack()

# Start the main loop
window.mainloop()
