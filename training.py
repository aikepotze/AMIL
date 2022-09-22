"""
Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
Training function
22-09-2022
"""


import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
from sklearn.metrics import roc_auc_score, average_precision_score
from csv import writer

from dataset import MILDataset
from network import Net, TruncatedLoss
from util import save


## Training setup
def train(
        full_dataset,
        save_path,
        result_df,

        # Experiment details
        model_name = 'model.pt',
        model_name_init = 'base.pt',
        results_name = 'results.csv',

        do_warm_start = False,
        do_only_test=False,
        do_save_model=True,

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        number_epochs = 3,
        batch_size = 16,
        lr = 1e-3,
        seed = 0,
        resnet_version = 'resnet50',

        # Class-level or image-level GT
        class_level_gt = True,

        # Decorrelation, 0.0 if not used, 10.0 standard if used
        decorrelation_weight = 0.0,

        # Simple MIL-max baseline
        do_mil_base = False,

        ## Methods for handling noisy attributes
        apply_only_to_positives = False, # Apply with MIL setting

        # Bootstrap
        dilute_predictions = 0.0,  # Bootstrap loss, set to 0 for no dilution
        dilute_negative_predictions = 0.0,  # Bootstrap loss, set to 0 for no dilution

        # Truncated loss
        do_truncated_loss = False,
        truncated_loss_q = 0.0, #0.5 is standard if used

        # Set threshold for class_attributes to be considered positive, 0-100
        positive_threshold = 10.0
        ):

    ## Training setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset, setup dataloaders
    data_train, data_test, class_attr_np, zsl_train_classes, zsl_test_classes, attr_names, rel_matrix = full_dataset.getData()

    dataset_train = MILDataset(data_train)
    dataset_test = MILDataset(data_test)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # Load the model
    n_attr = len(attr_names)
    if resnet_version == 'resnet50':
        resnet = torchvision.models.resnet50(pretrained=True)
    elif resnet_version == 'resnet101':
        resnet = torchvision.models.resnet101(pretrained=True)
    model = Net(resnet, n_attr)

    # Load the weights
    if do_only_test:
        number_epochs = 1
        do_warm_start = True
        do_save_model = False

    if do_warm_start:
        model.load_state_dict(torch.load(model_name_init, map_location=device), strict=True)

    # Load model and tensors to device
    model.to(device)
    rel_matrix = rel_matrix.to(device)
    class_attr = torch.FloatTensor(class_attr_np).to(device)

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam([{'params': list(model.parameters())[-2:], 'lr': lr, 'weight_decay': 0e-4},
                                  {'params': list(model.parameters())[0:-2], 'lr': lr*0.1}])

    loss_function = torch.nn.BCELoss(reduction='none')
    loss_truncated = TruncatedLoss(n_attr=n_attr, device=device, trainset_size=len(dataset_train), q=truncated_loss_q)

    # Define lists to store losses during training
    training_losses = []
    validation_losses = []


    for epoch in range(number_epochs):
        # Update learning rate
        for group in optimizer.param_groups:
            if epoch == 0:
                group['lr'] = group['lr']
            elif epoch == 1:
                group['lr'] = group['lr'] * 0.25
            elif epoch == 2:
                group['lr'] = group['lr'] * 0.25
            elif epoch == 3:
                group['lr'] = group['lr'] * 0.25
            elif epoch == 4:
                group['lr'] = group['lr'] * 0.25

        ################################################################################################################
        # Training phase:
        training_loss = 0
        training_decorrelation_loss = 0
        preds_train = []
        attr_train = []

        if do_only_test:
            model.eval()
        else:
            model.train()  # indicate to the network that we enter training mode
        pbar = tqdm(train_loader, position=0, leave=True)
        for i, sample in enumerate(pbar):

            # Load batch --- -------------------------------------------------------------------------------------------
            # Convert the inputs and GTs to torch Variable (will hold the computation
            # graph) and send them to the computation device (i.e. GPU).
            image = torch.FloatTensor(sample[0]).to(device)
            im_class = torch.LongTensor(sample[1]).to(device)
            idx = torch.LongTensor(sample[3]).to(device)

            # Load the sample attributes at class-level or at image-level
            if class_level_gt:
                attr = (class_attr[im_class, :] > positive_threshold).float()
            else:
                attr_scale = torch.FloatTensor(sample[2]).to(device)
                attr = (attr_scale > 0.5).float()


            ### Forward pass
            # Run the forward pass
            pred, maps, dist = model(image)
            pred = pred.sigmoid() # Move to within forward pass?

            # Bootstrap, dilute predictions with ground truth
            if dilute_predictions > 0:
                if apply_only_to_positives:
                    pred_for_boot = pred.detach()
                    weighted_attr = (attr == 0) * (dilute_negative_predictions) * (pred).detach() + (attr == 1) * (
                                (1 - dilute_predictions) * attr + (dilute_predictions) * pred_for_boot)
                else:
                    weighted_attr = ((1 - dilute_predictions) * attr + (dilute_predictions) * (pred).detach())
            else:
                weighted_attr = attr

            # Calculate loss -------------------------------------------------------------------------------------------
            ## Based on level of attribute ground truth, determine which attributes to enforce loss for each sample
            if class_level_gt:
                if do_mil_base:
                    _, topk_ids = (pred * (attr == 1)).topk(1, dim=0)
                    topk_mask = torch.nn.functional.one_hot(topk_ids, attr.shape[0]).sum(-1).to(device).float()
                    enforce = (attr == 0) + (attr == 1) * topk_mask  # attr.sum(0,keepdim=True)
                else:  # not do_mil
                    enforce = (attr == 0) + (attr == 1)
            else:  # not class_level_gt
                enforce = (attr == 0) + (attr == 1)



            ## Attribute loss
            # Trucated loss
            if do_truncated_loss:
                loss_truncated.update_weight(pred.cpu(), attr.cpu(), idx.cpu())
                loss_attr_gce = loss_truncated(pred, weighted_attr, idx)
                if apply_only_to_positives:
                    loss_attr = loss_function(pred, weighted_attr)
                    loss_attr = (attr == 0) * loss_attr + (attr == 1) * loss_attr_gce
                else:
                    loss_attr = loss_attr_gce
                loss_attr = loss_attr * enforce.detach()
            else:
                loss_attr = (-weighted_attr * (pred + 1e-16).log() - (1 - weighted_attr) * (
                            1 - pred + 1e-16).log()) * enforce.detach()

            loss_attr = loss_attr.mean()


            ## Decorrelation loss
            n_weight = model.fc.weight.squeeze().div(torch.norm(model.fc.weight.squeeze(), dim=1, keepdim=True) + 1e-6)
            loss_decorr = (((n_weight @ n_weight.T) - rel_matrix).abs() ** 2)
            loss_decorr = decorrelation_weight * loss_decorr.mean()


            ## Total loss
            loss = 1 * loss_attr + 1 * loss_decorr

            # Finish batch ---------------------------------------------------------------------------------------------
            # Backpropagate losses and update the optimizer
            if not do_only_test:
                loss.backward()
                optimizer.step()
                # We set the gradients of the model to 0.
                optimizer.zero_grad()

            # Save training losses, predictions and attributes
            training_loss += loss_attr.cpu().item()
            training_decorrelation_loss += loss_decorr.cpu().item()
            preds_train.append(pred.detach().cpu().numpy())
            attr_train.append(attr.cpu().numpy())

            # Update description of progress bar
            pbar.set_description(
                "Training loss attr: %f, decorr: %f" % (training_loss / (i + 1), training_decorrelation_loss / (i + 1)))

        # Finish training epoch ----------------------------------------------------------------------------------------
        print("At epoch #" + str(epoch) + ", loss = " + str(training_loss / len(train_loader)))

        # Store losses, predictions and attributes
        training_losses.append(training_loss / len(train_loader))
        preds_train = np.concatenate(preds_train)
        attr_train = np.concatenate(attr_train)

        # Optional, if you want to save your model:
        if do_save_model:
            torch.save(model.state_dict(), save_path + '/' + model_name + '.pt')
            print('Model saved\n')

        ################################################################################################################
        # Validation / testing phase:
        model.eval()

        validation_loss = 0
        correct = 0
        correct_per_class = 0
        preds = []
        scores = []
        attr_gts = []
        attr_gts_per_class = []
        im_classes = []

        # Select appropriate loader
        loader_to_use = test_loader

        with torch.no_grad():
            for i, sample in enumerate(tqdm(loader_to_use, position=0, leave=True)):
                # Load batch
                image = torch.FloatTensor(sample[0]).to(device)
                im_class = torch.LongTensor(sample[1]).to(device)
                attr = torch.FloatTensor(sample[2]).to(device)
                attr = (attr > 0.5).float()
                attr_per_class = (class_attr[im_class, :]).float()

                # Forward pass
                pred, _, _ = model(image)
                scores.append(pred.detach().cpu().numpy())
                pred = pred.sigmoid()

                # Calculate loss
                loss = loss_function(pred, attr)
                loss = loss.mean()

                # Store loss, class, attributes, predictions, number of correct predictions
                validation_loss += loss.cpu().item() / len(loader_to_use)
                im_classes.append(sample[1])
                attr_gts.append(attr.detach().cpu().numpy())
                attr_gts_per_class.append(attr_per_class.detach().cpu().numpy())

                preds.append(pred.detach().cpu().numpy())
                pred = pred > 0.5
                correct += pred.eq((attr).float().view_as(pred)).sum().item() / attr.shape[1]
                correct_per_class += pred.eq((attr_per_class.view_as(pred) > positive_threshold).float()).sum().item() / attr.shape[1]

        # Simple metrics -----------------------------------------------------------------------------------------------
        # Concatenate the stored results
        im_classes = np.concatenate(im_classes)
        attr_gts = np.concatenate(attr_gts)
        attr_gts_per_class = np.concatenate(attr_gts_per_class)
        preds = np.concatenate(preds)
        scores = np.concatenate(scores)


        attr_gts_4auc = np.float32(attr_gts)
        attr_gts_per_class_4auc = attr_gts_per_class > np.float32(positive_threshold)
        auc = roc_auc_score(attr_gts_4auc[:, attr_gts_4auc.var(0) != 0], preds[:, attr_gts_4auc.var(0) != 0],
                            average='macro')
        auc_per_class = roc_auc_score(attr_gts_per_class_4auc[:, attr_gts_per_class_4auc.var(0) != 0],
                                      preds[:, attr_gts_per_class_4auc.var(0) != 0], average='macro')
        validation_losses.append(validation_loss)

        ## Print results
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%), AUC: {:.4f}, Accuracy per class: ({:.2f}%), AUC per class: {:.4f}\n'.format(
                validation_loss, 100. * correct / len(loader_to_use.dataset), auc,
                                 100. * correct_per_class / len(loader_to_use.dataset), auc_per_class))
        print('AP:')
        print(average_precision_score(attr_gts_4auc[:, attr_gts_4auc.var(0) != 0], preds[:, attr_gts_4auc.var(0) != 0],
                                      average='macro'))

        # ZSL metrics --------------------------------------------------------------------------------------------------
        # DAP with row/column normalization
        class_attr_np0 = class_attr_np.copy() > np.float32(positive_threshold)
        test_ids_to_use = np.arange(len(im_classes))
        im_classes_to_use = im_classes[test_ids_to_use]
        preds_norm = np.concatenate((preds[test_ids_to_use, :], preds_train), axis=0)

        pa1 = class_attr_np0.mean(0, keepdims=True)*(1-2e-6) + 1e-6
        pa2 = preds_train.mean(0, keepdims=True)

        cl_votes_absence = np.exp(np.matmul(np.log((1 - preds_norm)  / ((1 - pa1) + 1e-16) + 1e-16), ((1 - class_attr_np0)/class_attr_np0.shape[1]).T))
        cl_votes_presence = np.exp(np.matmul(np.log(preds_norm / (pa1 + 1e-16) + 1e-16), ((class_attr_np0)/class_attr_np0.shape[1]).T))

        il_votes_absence = np.exp(np.matmul(np.log((1 - preds_norm) / ((1 - pa2) + 1e-16) + 1e-16), ((1 - class_attr_np0) / class_attr_np0.shape[1]).T))
        il_votes_presence = np.exp(np.matmul(np.log(preds_norm / (pa2 + 1e-16) + 1e-16), ((class_attr_np0) / class_attr_np0.shape[1]).T))

        votes_absence2 = np.exp(np.matmul(np.log((1 - preds_norm) ** 2 / ((1 - pa2)+ 1e-16) + 1e-16), ((1 - class_attr_np0)/class_attr_np0.shape[1]).T))

        ## DAP GZSL and ZSL metrics ------------------------------------------------------------------------------------
        cl_dap_votes = cl_votes_presence * cl_votes_absence
        cl_dap_votes = cl_dap_votes[0:len(test_ids_to_use), :]

        # GZSL
        seen_classes = np.setdiff1d(np.arange(cl_dap_votes.shape[1]), zsl_test_classes)
        mask_seen = np.array([b in zsl_train_classes for b in im_classes_to_use])
        seen_acc = (np.argmax(cl_dap_votes[mask_seen, :], axis=1) == im_classes_to_use[mask_seen]).mean()

        unseen_classes = zsl_test_classes
        mask_unseen = np.array([b in zsl_test_classes for b in im_classes_to_use])
        unseen_acc = (np.argmax(cl_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()

        h_acc = (2 * seen_acc * unseen_acc / (seen_acc + unseen_acc))

        print('GZSL s: %.4f' % seen_acc)
        print('GZSL u: %.4f' % unseen_acc)
        print('GZSL h: %.4f' % h_acc)

        # ZSL
        cl_dap_votes[:, np.setdiff1d(np.arange(cl_dap_votes.shape[1]), zsl_test_classes)] = -np.inf
        zsl_acc = (np.argmax(cl_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()
        print('ZSL: %.4f' % zsl_acc)


        ## Image level pa DAP GZSL and ZSL metrics ---------------------------------------------------------------------
        il_dap_votes = il_votes_presence * il_votes_absence

        il_dap_votes = il_dap_votes[0:len(test_ids_to_use), :]

        # GZSL
        seen_classes = np.setdiff1d(np.arange(il_dap_votes.shape[1]), zsl_test_classes)
        mask_seen = np.array([b in zsl_train_classes for b in im_classes_to_use])
        seen_acc_il_DAP = (np.argmax(il_dap_votes[mask_seen, :], axis=1) == im_classes_to_use[mask_seen]).mean()

        unseen_classes = zsl_test_classes
        mask_unseen = np.array([b in zsl_test_classes for b in im_classes_to_use])
        unseen_acc_il_DAP = (np.argmax(il_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()

        h_acc_il_DAP = (2 * seen_acc_il_DAP * unseen_acc_il_DAP / (seen_acc_il_DAP + unseen_acc_il_DAP))

        print('GZSL s: %.4f' % seen_acc_il_DAP)
        print('GZSL u: %.4f' % unseen_acc_il_DAP)
        print('GZSL h: %.4f' % h_acc_il_DAP)

        # ZSL
        il_dap_votes[:, np.setdiff1d(np.arange(il_dap_votes.shape[1]), zsl_test_classes)] = -np.inf
        zsl_acc_il_DAP = (np.argmax(il_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()
        print('ZSL: %.4f' % zsl_acc_il_DAP)

        ## DAP MIL GZSL metrics ----------------------------------------------------------------------------------------
        mil_dap_votes = il_votes_absence
        mil_dap_votes = mil_dap_votes[0:len(test_ids_to_use), :]

        # GZSL
        seen_classes = np.setdiff1d(np.arange(mil_dap_votes.shape[1]), zsl_test_classes)
        mask_seen = np.array([b in zsl_train_classes for b in im_classes_to_use])
        seen_acc_mil = (np.argmax(mil_dap_votes[mask_seen, :], axis=1) == im_classes_to_use[mask_seen]).mean()

        unseen_classes = zsl_test_classes
        mask_unseen = np.array([b in zsl_test_classes for b in im_classes_to_use])
        unseen_acc_mil = (np.argmax(mil_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()

        h_acc_mil = (2 * seen_acc_mil * unseen_acc_mil / (seen_acc_mil + unseen_acc_mil))

        print('GZSL s: %.4f' % seen_acc_mil)
        print('GZSL u: %.4f' % unseen_acc_mil)
        print('GZSL h: %.4f' % h_acc_mil)

        # ZSL
        mil_dap_votes[:, np.setdiff1d(np.arange(mil_dap_votes.shape[1]), zsl_test_classes)] = -np.inf
        zsl_acc_mil = (np.argmax(mil_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()
        print('ZSL: %.4f' % zsl_acc_mil)

        ## Squared DAP MIL GZSL metrics --------------------------------------------------------------------------------
        sq_mil_dap_votes = votes_absence2

        sq_mil_dap_votes = sq_mil_dap_votes[0:len(test_ids_to_use), :]

        # GZSL
        seen_classes = np.setdiff1d(np.arange(sq_mil_dap_votes.shape[1]), zsl_test_classes)
        mask_seen = np.array([b in zsl_train_classes for b in im_classes_to_use])
        seen_acc_sq_mil = (np.argmax(sq_mil_dap_votes[mask_seen, :], axis=1) == im_classes_to_use[mask_seen]).mean()

        unseen_classes = zsl_test_classes
        mask_unseen = np.array([b in zsl_test_classes for b in im_classes_to_use])
        unseen_acc_sq_mil = (
                    np.argmax(sq_mil_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()

        h_acc_sq_mil = (2 * seen_acc_sq_mil * unseen_acc_sq_mil / (seen_acc_sq_mil + unseen_acc_sq_mil))

        print('GZSL s: %.4f' % seen_acc_sq_mil)
        print('GZSL u: %.4f' % unseen_acc_sq_mil)
        print('GZSL h: %.4f' % h_acc_sq_mil)

        # ZSL
        sq_mil_dap_votes[:, np.setdiff1d(np.arange(sq_mil_dap_votes.shape[1]), zsl_test_classes)] = -np.inf
        zsl_acc_sq_mil = (np.argmax(sq_mil_dap_votes[mask_unseen, :], axis=1) == im_classes_to_use[mask_unseen]).mean()
        print('ZSL: %.4f' % zsl_acc_sq_mil)

        # Trait metrics ------------------------------------------------------------------------------------------------
        #zsl_train_classes = np.setdiff1d(np.arange(votes.shape[1]), zsl_test_classes)
        is_train = [c in zsl_train_classes for c in im_classes]
        is_test = [c in zsl_test_classes for c in im_classes]

        # Calculate AP and AUC for seen (train), unseen (test) and harmonic means
        attr_gts_4ap = attr_gts_4auc[:, (attr_gts_4auc[is_train, :].var(0) != 0)]
        preds_4ap = preds[:, (attr_gts_4auc[is_train, :].var(0) != 0)]
        ap_train = average_precision_score(attr_gts_4ap[is_train, :], preds_4ap[is_train, :], average='macro')
        auc_train = roc_auc_score(attr_gts_4ap[is_train, :], preds_4ap[is_train, :], average='macro')

        attr_gts_4ap = attr_gts_4auc[:, (attr_gts_4auc[is_test, :].var(0) != 0)]
        preds_4ap = preds[:, (attr_gts_4auc[is_test, :].var(0) != 0)]
        ap_test = average_precision_score(attr_gts_4ap[is_test, :], preds_4ap[is_test, :], average='macro')
        auc_test = roc_auc_score(attr_gts_4ap[is_test, :], preds_4ap[is_test, :], average='macro')

        ap_h = (2 * ap_train * ap_test / (ap_train + ap_test))
        auc_h = (2 * auc_train * auc_test / (auc_train + auc_test))

        print('AP train: %.4f' % ap_train)
        print('AP test: %.4f' % ap_test)
        print('AP h: %.4f' % ap_h)
        print('AUC train: %.4f' % auc_train)
        print('AUC test: %.4f' % auc_test)
        print('AUC h: %.4f' % auc_h)
        
        # Calculate AP and AUC with respect to the class-level labels on the test set
        attr_gts_4ap_per_class = attr_gts_per_class_4auc[:,(attr_gts_per_class_4auc[is_test,:].var(0)>0)]
        preds_4ap = preds[:,(attr_gts_per_class_4auc[is_test,:].var(0)!=0)]
        ap_test_per_class = average_precision_score(attr_gts_4ap_per_class[is_test, :], preds_4ap[is_test,:], average='macro')
        auc_test_per_class = roc_auc_score(attr_gts_4ap_per_class[is_test, :], preds_4ap[is_test,:], average='macro')

        print('AP test per class: %.4f' % ap_test_per_class)
        print('AUC test per class: %.4f' % auc_test_per_class)

    ## Per-attribute metrics ---------------------------------------------------------------------------------------
    def get_ap_auc_per_class(attr_gts_4auc, preds, mask):
        """From arrays of attribute ground truth and predictions, calculate AP and AUC per class."""
        preds_per_attr = preds[mask, :]
        attr_gts_per_attr = attr_gts_4auc[mask, :]

        # Calculate AP and AUC per class for seen classes
        ap_per_attr = []
        auc_per_attr = []
        for i in range(preds.shape[1]):
            # If class is empty, log i in empty_classes
            if attr_gts_per_attr[:, i].var() == 0:
                ap_per_attr.append(-1)
                auc_per_attr.append(-1)
            else:
                ap_per_attr.append(average_precision_score(attr_gts_per_attr[:, i], preds_per_attr[:, i], average='macro'))
                auc_per_attr.append(roc_auc_score(attr_gts_per_attr[:, i], preds_per_attr[:, i], average='macro'))
        return ap_per_attr, auc_per_attr

    def write_to_csv(tgt_list, tgt_csv_name):
        """Writes list to csv file"""
        # Store ap and auc
        #if csv file doesn't exist, create it
        csv_path = os.path.join(save_path, tgt_csv_name)

        #if csv file doesn't exist, create it with header
        if not os.path.isfile(csv_path):
            # Create empty csv file
            with open(csv_path, 'w') as csvfile:
                csv_writer = writer(csvfile)
                csv_writer.writerow(['model'] + list(range(preds.shape[1])))

        # Write list to csv file
        with open(csv_path, 'a+', newline='') as csv:
            # Create a writer object from csv module
            csv_writer = writer(csv)
            # Add contents of list as last row in the csv file
            csv_writer.writerow([model_name] + tgt_list)

    # Calculate AP and AUC per class for seen classes
    seen_ap_per_attr, seen_auc_per_attr = get_ap_auc_per_class(attr_gts_4auc, preds, is_train)
    write_to_csv(seen_ap_per_attr, 'seen_ap_per_attr.csv')
    write_to_csv(seen_auc_per_attr, 'seen_auc_per_attr.csv')

    # Calculate AP and AUC per class for unseen classes
    unseen_ap_per_attr, unseen_auc_per_attr = get_ap_auc_per_class(attr_gts_4auc, preds, is_test)
    write_to_csv(unseen_ap_per_attr, 'unseen_ap_per_attr.csv')
    write_to_csv(unseen_auc_per_attr, 'unseen_auc_per_attr.csv')

    run_results = [model_name,
                    100*ap_train, 100*ap_test, 100*ap_h,
                    100*auc_train, 100*auc_test, 100*auc_h,
                    100*ap_test_per_class, 100*auc_test_per_class,
                    100*seen_acc, 100*unseen_acc, 100*h_acc, 100*zsl_acc,
                    100 * seen_acc_il_DAP, 100 * unseen_acc_il_DAP, 100 * h_acc_il_DAP, 100 * zsl_acc_il_DAP,
                    100*seen_acc_mil, 100*unseen_acc_mil, 100*h_acc_mil, 100*zsl_acc_mil,
                    100 * seen_acc_sq_mil, 100 * unseen_acc_sq_mil, 100 * h_acc_sq_mil, 100 * zsl_acc_sq_mil,
                    dilute_predictions, decorrelation_weight, truncated_loss_q]

    # Save results and return dataframe
    return save(result_df, run_results, save_path, results_name)
