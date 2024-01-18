from torch import nn

class GE2E_Loss(nn.Module):

    def __init__(self, weight=1, bias=1, loss='contrast',is_Normalize=False) -> None:

        ''' 
            Here we are try to implement GE2E contrast loss function

            Accept input is size (N,M,D):
                where N is the number of speaker in the batch,
                M is the number of utterence per speaker,
                each utterance is of size D 

            Args:
                - init_w (float) : define inital weight
                - init_b (float) : define inital bias 

            Here w,b are learnable parameters which can be learned by back propgation algorithm.
        '''

        super(GE2E_Loss, self).__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.bias = nn.Parameter(torch.tensor(bias))

        if loss != 'contrast':
            raise ValueError(f"{loss} not avaliable")

        self.is_Normalize = is_Normalize
        
        self.embed_loss = self.contrast_loss




    def calculate_new_centroid(self, embeddings, speaker_id, utterance_id,centroids):

        """ 
            This is an helper function that try to caclulate modified centroid by skipping, a particular utterance for 
            stable convergence.
        """

        return_centroids = []

        stable_centroid = torch.cat((embeddings[speaker_id, :utterance_id], embeddings[speaker_id, utterance_id+1:]))

        for centroid_id, centroid in enumerate(centroids):
            if speaker_id == centroid_id:
                return_centroids.append(stable_centroid)
            else:
                return_centroids.append(centroid)

        return torch.stack(return_centroids)
    



    def calculate_similarity(self, embeddings, centroids):
        """
            Calculate the cosine similarity between embeddings vector and speakers centroid. 
            It finally returns a tensor of shape N x M x N, where any entry say tensor[j,i,k] 
            represents the cosine similarity between speaker j utterance i and speaker k centroid.

            Args:
                embeddings: embeddings of the utterances of shape N x M x D.
                centroids: centroid of each speaker of shape N x D.
        """

        return_similarity_matrix = []

        for speaker_id, speaker in enumerate(embeddings):
            row_wise_similarity = []
            for utterance_id, utterance in enumerate(speaker):

                centroid = self.calculate_new_centroid(embeddings, speaker_id, utterance_id, centroids)

                similarity_vector = nn.functional.cosine_similarity(utterance, centroid)

                row_wise_similarity.append(torch.cat(similarity_vector))

            return_similarity_matrix.append(torch.cat(row_wise_similarity))

        return torch.cat(return_similarity_matrix)

    


    def contrast_loss(self, similarity_matrix, N, M):
        """
            This function is try to compute the loss using contrast loss function. 
            Return an matrix of shape N x M, and j,i entry of it represent loss 
            for speaker j utterance i. 

            Args: 
                similarity_matrix : Similarity matrix of shape N x M x N.
                N : # of speakers
                M : # of utterance 
        """

        return_loss = []
        sigmoid_similarity = torch.sigmoid(similarity_matrix)
        for speaker_id in range(1,N+1):
            speaker_loss = []
            for utterance_id in range(1,M+1):
                poisitive_component = 1. - sigmoid_similarity[speaker_id, utterance_id, speaker_id]
                aggresive_negative_component = torch.max(torch.cat((sigmoid_similarity[speaker_id, utterance_id,:speaker_id],sigmoid_similarity[speaker_id, utterance_id,speaker_id+1:])))
                speaker_loss.append(poisitive_component + aggresive_negative_component)
            return_loss.append(torch.stack(speaker_loss))

        return torch.stack(return_loss)
                


                
    def forward(self, embeddings): 
        
        """ 
            This loss function is try to implement forward pass using contrast loss function.

            Args:
                embeddings : This is a tensor of shape N x M x D, dimensional matrix where,
                             N : # of speakers, M : # of utterance, D : dimension of each utterance
        
        """

        N,M,D = embeddings.shape

        if self.is_Normalize:
            embeddings = nn.functional.normalize(embeddings, p=2,dim=-1)

        centroids = torch.mean(embeddings, 1) # N x D

        cosine_similarity = self.calculate_similarity(embeddings, centroids) # N x M x N

        similarity_matrix = cosine_similarity * self.weight + self.bias
        Loss = self.contrast_loss(similarity_matrix, N,M)

        return Loss.sum()
