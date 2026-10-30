import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# GRAPHAUTOENC: uses a graph-based encoder operating on the item-item co-purchase graph

class GraphEncoderVisual2Textual(nn.Module):
    def __init__(self):
        super(GraphEncoderVisual2Textual, self).__init__()
        self.conv1 = GCNConv(2048, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

    def message_passing(self, x, edge_index):
        encoded = self.conv1(x, edge_index)
        encoded = nn.functional.leaky_relu(encoded)
        encoded = self.conv2(encoded, edge_index)
        encoded = nn.functional.leaky_relu(encoded)
        return encoded

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
    
class GraphEncoderTextual2Visual(nn.Module):
    def __init__(self):
        super(GraphEncoderTextual2Visual, self).__init__()
        self.conv1 = GCNConv(768, 1024)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

    def message_passing(self, x, edge_index):
        encoded = self.conv1(x, edge_index)
        encoded = nn.functional.leaky_relu(encoded)
        return encoded

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

# Child class of ImputatorFather
class GraphautoencImputator(ImputatorFather):

    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, train_path, top_k, seed,
                missing_percentage_visual, missing_percentage_textual, items_file,  num_epochs, learning_rate, batch_size):
        
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_visual_file = missing_visual_file
        self.missing_textual_file = missing_textual_file
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.train_path = train_path
        self.top_k = top_k
        self.seed = seed   
        self.missing_percentage_visual = missing_percentage_visual
        self.missing_percentage_textual = missing_percentage_textual
        self.items_file = items_file
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the graph autoencoder to reconstruct textual from visual
    def _train_graph_autoencoder_textual(self, visual_normalized, textual_normalized, adj):   
        
        model = GraphEncoderVisual2Textual()
        model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)           # default: lr=1e-4
        items = torch.arange(visual_normalized.shape[0])
        dataset = TensorDataset(items, textual_normalized)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  # default: batch_size=32

        cumulative_loss = 0        
        for epoch in range(self.num_epochs):                                        # default: num_epochs=1000
            for batch in dataloader:
                features = model.message_passing(visual_normalized, adj.to(self.device))
                items_id, targets = batch
                outputs = model.forward(features[items_id].to(self.device))
                loss = criterion(outputs, targets)
                cumulative_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % (self.num_epochs // 10) == 0:
                avg_loss_per_epoch = cumulative_loss / (self.num_epochs // 10)
                print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss_per_epoch:.4f}")
                cumulative_loss = 0

        return model

    # Train the graph autoencoder to reconstruct visual from textual
    def _train_graph_autoencoder_visual(self, textual_normalized, visual_normalized, adj):  
    
        model = GraphEncoderTextual2Visual()
        model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        items = torch.arange(visual_normalized.shape[0])
        dataset = TensorDataset(items, visual_normalized)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        cumulative_loss = 0         
        for epoch in range(self.num_epochs):
            for batch in dataloader:
                features = model.message_passing(textual_normalized, adj.to(self.device))
                items_id, targets = batch
                outputs = model.forward(features[items_id].to(self.device))
                loss = criterion(outputs, targets)
                cumulative_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % (self.num_epochs // 10) == 0:
                avg_loss_per_epoch = cumulative_loss / (self.num_epochs // 10)
                print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss_per_epoch:.4f}")
                cumulative_loss = 0

        return model

    # Impute textual missing modality
    def _impute_textual_modality(self, missing_textual_indexed, missing_visual_indexed, 
                                 visual_embeddings_folder_indexed, all_present, train_indexed_path,
                                 model_impute_textual, visual_normalized, visual_min, visual_max,
                                 textual_min, textual_max, textual_global_mean, items_map, output_path):
        
        for miss in missing_textual_indexed:
            if miss in missing_visual_indexed:
                output = (textual_global_mean * (textual_max - textual_min)) + textual_min
                miss_original = self.inverse_mapping(items_map, miss)
                np.save(os.path.join(output_path, f'{miss_original}.npy'), output.detach().cpu().numpy())
            else:
                current_all_present = all_present + [miss]
                inner_map_item = {it: idx for idx, it in enumerate(current_all_present)}
                train = pd.read_csv(train_indexed_path, sep='\t', header=None)
                train = train[train[1].isin(current_all_present)]
                inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}
                train[0] = train[0].map(inner_map_user)
                train[1] = train[1].map(inner_map_item)
                adj = self.get_item_item(train, num_items=len(current_all_present))
 
                visual_input = torch.from_numpy(np.load(os.path.join(visual_embeddings_folder_indexed, f'{miss}.npy')))
                visual_input = (visual_input - visual_min.cpu()) / (visual_max.cpu() - visual_min.cpu() + 1e-8)
                visual_input = visual_input.to(self.device)
                features = model_impute_textual.message_passing(
                    torch.concat((visual_normalized, visual_input), dim=0).to(self.device), 
                    adj.to(self.device)
                )
                output = model_impute_textual.forward(features[-1].to(self.device))
                output = (output * (textual_max - textual_min)) + textual_min
                miss_original = self.inverse_mapping(items_map, miss)
                np.save(os.path.join(output_path, f'{miss_original}.npy'), output.detach().cpu().numpy())

    # Impute visual missing modality
    def _impute_visual_modality(self, missing_visual_indexed, missing_textual_indexed,
                                textual_embeddings_folder_indexed, all_present, train_indexed_path,
                                model_impute_visual, textual_normalized, textual_min, textual_max,
                                visual_min, visual_max, visual_global_mean, items_map, output_path):
        
        for miss in missing_visual_indexed:
            if miss in missing_textual_indexed:
                output = (visual_global_mean * (visual_max - visual_min)) + visual_min
                miss_original = self.inverse_mapping(items_map, miss)
                np.save(os.path.join(output_path, f'{miss_original}.npy'), output.detach().cpu().numpy())
            else:
                current_all_present = all_present + [miss]
                inner_map_item = {it: idx for idx, it in enumerate(current_all_present)}
                train = pd.read_csv(train_indexed_path, sep='\t', header=None)
                train = train[train[1].isin(current_all_present)]
                inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}
                train[0] = train[0].map(inner_map_user)
                train[1] = train[1].map(inner_map_item)
                adj = self.get_item_item(train, num_items=len(current_all_present))

                textual_input = torch.from_numpy(np.load(os.path.join(textual_embeddings_folder_indexed, f'{miss}.npy')))
                textual_input = (textual_input - textual_min.cpu()) / (textual_max.cpu() - textual_min.cpu() + 1e-8)
                textual_input = textual_input.to(self.device)
                features = model_impute_visual.message_passing(
                    torch.concat((textual_normalized, textual_input), dim=0).to(self.device), 
                    adj.to(self.device)
                )
                output = model_impute_visual.forward(features[-1].to(self.device))
                output = (output * (visual_max - visual_min)) + visual_min 
                miss_original = self.inverse_mapping(items_map, miss)
                np.save(os.path.join(output_path, f'{miss_original}.npy'), output.detach().cpu().numpy())

    def impute(self):

        self.set_seed()

        output_visual = self.output_paths[0] + '/graphautoenc'
        os.makedirs(output_visual, exist_ok=True)
        output_textual = self.output_paths[1] + '/graphautoenc'
        os.makedirs(output_textual, exist_ok=True)

        if self.missing_percentage_visual is not None:
            errors = {'visual': [], 'textual': []}
            
            for percentage_visual, percentage_textual in zip(self.missing_percentage_visual, self.missing_percentage_textual):
              try:  
                # Create artificial missing items and folders
                missing_visual_items = self.compute_artificial_missing_items(self.dataset_path, percentage_visual, self.items_file, self.seed, file_name=f"artificial_missing_visual_file_{percentage_visual}_graphautoenc")
                result_path_percentage_visual = os.path.join(output_visual, f"{percentage_visual}")
                os.makedirs(result_path_percentage_visual, exist_ok=True)   
                missing_visual_npy = [f'{item}.npy' for item in missing_visual_items]
                visual_folder = f"{self.dataset_path}/non_masked_visual_folder_{percentage_visual}"
                self.create_non_masked_folder(self.input_paths[0], missing_visual_npy, visual_folder)
                
                missing_textual_items = self.compute_artificial_missing_items(self.dataset_path, percentage_textual, self.items_file, self.seed+1, file_name=f"artificial_missing_textual_file_{percentage_textual}_graphautoenc")
                result_path_percentage_textual = os.path.join(output_textual, f"{percentage_textual}")
                os.makedirs(result_path_percentage_textual, exist_ok=True)    
                missing_textual_npy = [f'{item}.npy' for item in missing_textual_items]
                textual_folder = f"{self.dataset_path}/non_masked_textual_folder_{percentage_textual}"
                self.create_non_masked_folder(self.input_paths[1], missing_textual_npy, textual_folder)

                # Mapping and indexing
                visual_embeddings_folder_indexed, missing_visual_indexed, items_map, train_indexed_path = self.mapping(self.dataset_path, self.train_path, visual_folder, missing_visual_items, modality='visual')
                missing_visual_indexed = set(pd.read_csv(missing_visual_indexed, sep='\t', header=None)[0].to_list())
                textual_embeddings_folder_indexed, missing_textual_indexed, _, _ = self.mapping(self.dataset_path, self.train_path, textual_folder, missing_textual_items, modality='textual')
                missing_textual_indexed = set(pd.read_csv(missing_textual_indexed, sep='\t', header=None)[0].to_list())

                # Load features
                (visual_features, textual_features, visual_shape, textual_shape, all_present, inner_map_item) = self.load_indexed_features(visual_embeddings_folder_indexed, textual_embeddings_folder_indexed, missing_visual_indexed, missing_textual_indexed)
                # Prepare adjacency matrix
                adj = self.prepare_adjacency_matrix(train_indexed_path, all_present)
                # Normalize
                (visual_normalized, textual_normalized, visual_min, visual_max, textual_min, textual_max, visual_global_mean, textual_global_mean) = self.normalize_features(visual_features, textual_features)
                # Train models
                model_impute_textual = self._train_graph_autoencoder_textual(visual_normalized, textual_normalized, adj)
                model_impute_visual = self._train_graph_autoencoder_visual(textual_normalized, visual_normalized, adj)
                # Impute
                self._impute_textual_modality(missing_textual_indexed, missing_visual_indexed, visual_embeddings_folder_indexed, all_present, train_indexed_path, model_impute_textual, visual_normalized,
                                              visual_min, visual_max, textual_min, textual_max, textual_global_mean, items_map, result_path_percentage_textual)
                self._impute_visual_modality(missing_visual_indexed, missing_textual_indexed, textual_embeddings_folder_indexed, all_present, train_indexed_path, model_impute_visual, textual_normalized,
                                             textual_min, textual_max, visual_min, visual_max, visual_global_mean, items_map, result_path_percentage_visual)

                # Compute reconstruction error
                mse_visual, cosine_sim_visual = self.compute_reconstruction_errors_for_modality(self.input_paths[0], result_path_percentage_visual, missing_visual_npy)
                errors['visual'].append((percentage_visual, mse_visual, cosine_sim_visual))
                mse_textual, cosine_sim_textual = self.compute_reconstruction_errors_for_modality(self.input_paths[1], result_path_percentage_textual, missing_textual_npy)
                errors['textual'].append((percentage_textual, mse_textual, cosine_sim_textual))
            
              finally: 
                    self.cleanup_temp_resources([visual_folder, textual_folder, f"{visual_embeddings_folder_indexed}", f"{textual_embeddings_folder_indexed}", f"{missing_visual_indexed}", f"{missing_textual_indexed}", f"{train_indexed_path}"])

            return errors
        
        else:
            
            missing_visual_items = self.compute_missing_items(self.dataset_path, self.missing_visual_file)
            missing_textual_items = self.compute_missing_items(self.dataset_path, self.missing_textual_file)
            visual_folder = self.input_paths[0]
            textual_folder = self.input_paths[1]

            # Mapping and indexing
            visual_embeddings_folder_indexed, missing_visual_indexed, items_map, train_indexed_path = self.mapping(self.dataset_path, self.train_path, visual_folder, missing_visual_items, modality='visual')
            missing_visual_indexed = set(pd.read_csv(missing_visual_indexed, sep='\t', header=None)[0].to_list())
            textual_embeddings_folder_indexed, missing_textual_indexed, _, _ = self.mapping(self.dataset_path, self.train_path, textual_folder, missing_textual_items, modality='textual')
            missing_textual_indexed = set(pd.read_csv(missing_textual_indexed, sep='\t', header=None)[0].to_list())

            # Load features
            (visual_features, textual_features, visual_shape, textual_shape, all_present, inner_map_item) = self.load_indexed_features(visual_embeddings_folder_indexed, textual_embeddings_folder_indexed, missing_visual_indexed, missing_textual_indexed)
            # Prepare adjacency matrix
            adj = self.prepare_adjacency_matrix(train_indexed_path, all_present)
            # Normalize
            (visual_normalized, textual_normalized, visual_min, visual_max, textual_min, textual_max, visual_global_mean, textual_global_mean) = self.normalize_features(visual_features, textual_features)
            # Train models
            model_impute_textual = self._train_graph_autoencoder_textual(visual_normalized, textual_normalized, adj)
            model_impute_visual = self._train_graph_autoencoder_visual(textual_normalized, visual_normalized, adj)
            # Impute
            self._impute_textual_modality(missing_textual_indexed, missing_visual_indexed, visual_embeddings_folder_indexed, all_present, train_indexed_path, model_impute_textual, visual_normalized,
                                          visual_min, visual_max, textual_min, textual_max, textual_global_mean, items_map, output_textual)
            self._impute_visual_modality(missing_visual_indexed, missing_textual_indexed, textual_embeddings_folder_indexed, all_present, train_indexed_path, model_impute_visual, textual_normalized,
                                         textual_min, textual_max, visual_min, visual_max, visual_global_mean, items_map, output_visual)
