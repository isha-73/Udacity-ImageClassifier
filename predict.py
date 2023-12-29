import image_utilities, network_utilities
import argparse , torch , json

def make_predictions(model,topk,gpu,category_names,img):
    
    model.eval()
    class_to_idx = model.class_to_idx
    idx_to_class = {class_to_idx[k]: k for k in class_to_idx}
    # reference from jupiter notebook
    img= torch.from_numpy(img).float()
    img = torch.unsqueeze(img, dim=0)

    if gpu and torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()
        

    with torch.no_grad():
        op = model.forward(img)
        pred = torch.exp(op).topk(topk)

    # Moving results to  CPU 
    prob = pred[0][0].cpu().data.numpy().tolist()
    classes = pred[1][0].cpu().data.numpy()
    
    classes = [idx_to_class[i] for i in classes] #iterating over each index
    
    if category_names != '': # not null
        
        with open(category_names, 'r') as f: # reading all categories
            cat_to_name = json.load(f)
            
        classes = [cat_to_name[x] for x in classes]

    return prob, classes # probabilities and classes predicted


def main():
    parser = argparse.ArgumentParser(description="Using Pre-trained Deep Learning Model for Flower Classification")
    
    parser.add_argument('input', type=str, help="image")
    parser.add_argument('checkpoint', type=str, help='path - pre-trained model')
    parser.add_argument('--top_k', default=4, type=int)
    parser.add_argument('--category_names', default='', type=str)
    parser.add_argument('--gpu', default=True, action='store_true')
    args =parser.parse_args()
    
    proc_img = image_utilities.processimage(args.input)
    model = network_utilities.load_model(args.checkpoint) # loading model from checkpoint
    
    probs, classes = make_predictions( model, args.top_k, args.gpu, args.category_names,proc_img)
    print(f"{args.top_k} predictions which are matching: {list(zip(classes, probs))}")


if __name__ == '__main__':
    main()
