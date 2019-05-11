import torch
from torch.optim import Adam, SGD

class Trainer:
    '''Class for model training'''

    def __init__(self, model, optimizer, criterion, mutual_info_loss, clip_norm,
                 writer, num_updates, device, multi_gpu):

        self.device = device
        self.model = model
        self.model_optimizer = optimizer
        self.criterion = criterion
        self.mutual_info_loss = mutual_info_loss
        self.clip_norm = clip_norm
        self.writer = writer

        self.classifier = torch.nn.Linear(512, 10)
        self.classifier.to(device)
        if multi_gpu:
            self.classifier = torch.nn.DataParallel(self.classifier)
        self.classifier_optimizer = SGD(self.classifier.parameters(), lr=0.25)

        self.num_updates = num_updates

    def train_step(self, batch):
        images, labels = batch

        mi_loss, vectors = self.forward_mi(images, train=True)
        self.backward_mi(mi_loss)

        vectors = vectors.detach()
        classification_loss, score = self.forward_classifier(vectors, labels, train=True)
        self.backward_classifier(classification_loss)

        return mi_loss, classification_loss, score

    def test_step(self, batch):
        images, labels = batch

        mi_loss, vectors = self.forward_mi(images, train=False)
        classification_loss, score = self.forward_classifier(vectors, labels, train=False)

        return mi_loss, classification_loss, score

    def forward_mi(self, images, train):
        if train:
            self.model.train()
            self.model_optimizer.zero_grad()
            self.num_updates += 1
        else:
            self.model.eval()

        features, vectors = self.model(images)
        mi_loss = self.mutual_info_loss(features, vectors, measure='fd', mode='nce')

        return mi_loss, vectors

    def forward_classifier(self, vectors, labels, train):
        if train:
            self.classifier.train()
            self.classifier_optimizer.zero_grad()
        else:
            self.classifier.eval()

        probs = self.classifier(vectors)
        classification_loss = self.criterion(probs, labels)

        preds = torch.argmax(probs, dim=1).cpu()
        labels = labels.cpu()
        score = (preds == labels).sum().float()/len(preds)

        return classification_loss, score

    def backward_mi(self, loss):
        loss.backward()
        self.model_optimizer.step()

    def backward_classifier(self, loss):
        loss.backward()
        self.classifier_optimizer.step()
