import argparse, time
import data_utilities , network_utilities
import torch, torch.nn as nn


def train_epoch(model, train_loader, valid_loader, criteria, optimizer, gpu):
    
    if gpu and torch.cuda.is_available():
         model.cuda()

    print("Training...")
    model.train()
    
    for img, labels in train_loader:
        
        if gpu and torch.cuda.is_available():
            img, labels = img.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        op = model(img)
        
        minus= criteria(op, labels)
        minus.backward()
        optimizer.step()

    print("Validation...")
    model.eval()
    with torch.no_grad():
        train_loss, train_accuracy = make_valid(train_loader, criteria, gpu,model)
        valid_loss, valid_accuracy = make_valid(valid_loader, criteria, gpu,model)

    return train_loss, train_accuracy, valid_loss, valid_accuracy

def make_valid( data, criteria, gpu,model):
    if gpu and torch.cuda.is_available():
        model.to('cuda')

    sub = 0
    acc = 0

    for img, labels in data:
        if gpu and torch.cuda.is_available():
            img, labels = img.to('cuda'), labels.to('cuda')

        op = model(img)
        sub += criteria(op, labels) #loss
        pred = torch.exp(op).data 
        similarity = (labels.data == pred.max(1)[1])
        acc += similarity.type_as(torch.FloatTensor()).mean()

    sub /= len(data)
    acc /= len(data)

    return sub, acc # to return loss and accuracy

def main():
    
    parse = argparse.ArgumentParser(description="Training Model of Flower Classification")
    parse.add_argument('data_dir', type=str, help="required data directory*** ")
    parse.add_argument('--arch', default='vgg16', help='Deep NN architecture: vgg16')
#     parse.add_argument('--save_dir', default='', type=str, help="directory of checkpoints")
    
    parse.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parse.add_argument('--hidden_units', default=256, type=int, help='number of neurons in the hidden layer')
    parse.add_argument('--output_units', default=102, type=int, help='number of output categories')
    parse.add_argument('--drop_prob', default=0.1, type=float, help='dropout probability')
    parse.add_argument('--epochs', default=5, type=int, help='number of epochs for training')
    parse.add_argument('--gpu', default=True, action='store_true', help='use GPU for training?')
    
    args = parse.parse_args()
    print(f"{args.arch} architecture")

    train_loader, valid_loader, _, class_to_idx = data_utilities.data_loaders(args.data_dir)
    model = network_utilities.build_network(args.arch, args.output_units,args.hidden_units, args.drop_prob)
    model.class_to_idx = class_to_idx
    criteria = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), args.learning_rate)

    for epoch in range(args.epochs):
        train_loss, train_acc, valid_loss, valid_acc = train_epoch(model, train_loader, valid_loader, criteria, optimizer, args.gpu)
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    network_utilities.save_model(model, args.arch, args.epochs, args.learning_rate, args.hidden_units)


if __name__ == '__main__':
    main()
