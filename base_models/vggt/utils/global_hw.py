global_h = 28
global_w = 37
is_acc = False

def update_var(new_h, new_w):
    global global_h
    global global_w
    global_h = new_h
    global_w = new_w
    
def update_acc_true():
    global is_acc
    is_acc = True
    
def update_acc_false():
    global is_acc
    is_acc = False