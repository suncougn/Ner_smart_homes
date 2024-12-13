from tqdm import tqdm
import torch 
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat==labels_flat)/ len(labels_flat)

def compute_metrics(pred, labels, label_map):
    pred_flat = np.argmax(pred, axis=2)
    labels_flat = labels
    pred_tags = [[label_map[p] for p in pred_seq] for pred_seq in pred_flat]
    label_tags = [[label_map[l] for l in label_seq] for label_seq in labels_flat]
    precision = precision_score(label_tags, pred_tags)
    recall = recall_score(label_tags, pred_tags)
    f1 = f1_score(label_tags, pred_tags)

    exact_match = np.mean(
        [np.array_equal(pred_tags[i], label_tags[i]) for i in range(len(label_tags))]
    )
    accuracy = np.sum(pred_flat.flatten() == labels_flat.flatten()) / len(labels_flat.flatten())
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Exact Match (EM)": exact_match
    }

class trainer():
    def train(self, model, dataloader, epoch, epochs, writer, criterion, optimizer, scheduler, device, length, max_grad_norm):
        progress_bar = tqdm(dataloader, colour = '#800080', ncols = 120)
        total_loss = 0
        total_samples = 0
        b_errors = 0
        model.train()
        for iter, batch in enumerate(progress_bar):
            b_input_ids, b_attention_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask
            )
            loss = criterion(outputs.view(-1, length), b_labels.view(-1).long())
            loss.backward()
            total_loss+=loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_samples+=b_input_ids.size(0)
            progress_bar.set_description(f"TRAIN | Epoch: {epoch+1}/{epochs} | Iter: {iter+1}/{len(dataloader)} | Error: {b_errors}/{len(dataloader)} | Loss: {(total_loss/total_samples):.4f}")
        writer.add_scalar('Train/Loss', total_loss/total_samples, epoch+1)
    def validation(self, model, dataloader, criterion, device, length):
        model.eval()
        eval_loss, eval_acc, nb_eval_steps = 0, 0, 0
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask
                )
            loss = criterion(outputs.view(-1, length), b_labels.view(-1).long())
            logits = outputs.detach().cpu().numpy()
            labels_ids = b_labels.to('cpu').numpy()
            eval_loss += loss.mean().item()
            eval_acc +=flat_accuracy(logits, labels_ids)
            nb_eval_steps += 1
        return eval_loss/nb_eval_steps, eval_acc/nb_eval_steps
    def evaluate_model(self, model , dataloader, label_map, device):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                ib_input_ids, b_attention_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask
                )
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics(all_preds, all_labels, label_map)
        return metrics
if __name__ == "__main__":
    pass


'''
## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids)

        # get the loss
        loss = criterion(outputs.view(-1, len(tag2idx)), b_labels.view(-1).long())
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids)
        loss = criterion(outputs.view(-1, len(tag2idx)), b_labels.view(-1).long())
        # Move logits and labels to CPU
        logits = outputs.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += loss.mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print()
'''