import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



class density_map(object):
    def __init__(self,data,x_col,y_col,class_col,
                resize_param=None,
                window_size=None,
                padding=None,
                stride=None):
        self.data=data
        self.x_col=x_col
        self.y_col=y_col
        self.class_col=class_col
        self.resize_param=resize_param if resize_param is not None else (500,500)
        self.window_size=window_size if window_size is not None else (9,9)
        self.padding=padding if padding is not None else 0.1
        self.stride=stride if stride is not None else 1
        return

    def get_data_bounds(self,x,y):
        """
        if x==None and y==None:
            x_lims=(min(self.data[self.x_col]),max(self.data[self.x_col]))
            y_lims=(min(self.data[self.y_col]),max(self.data[self.y_col]))
            return x_lims,y_lims
        """
        #print(x)
        x_lims=(min(x),max(x))
        y_lims=(min(y),max(y))
        return x_lims,y_lims

    def init_output_map(self,x,y):
        #print(x)

        x_lims,y_lims=self.get_data_bounds(x,y)
        x_lims,y_lims=self.get_padded_bounds(x_lims,y_lims)

        map_len_x=np.ceil(x_lims[1]-x_lims[0])
        map_len_y=np.ceil(y_lims[1]-y_lims[0])

        map_len_x=int(np.floor(((map_len_x-self.window_size[0])/self.stride)) + 1)
        map_len_y=int(np.floor(((map_len_y-self.window_size[1])/self.stride)) + 1)

        return np.zeros(shape=(map_len_y,map_len_x))

    def init_window(self,x,y):
        x_lims,y_lims=self.get_data_bounds(x,y)
        x_lims,y_lims=self.get_padded_bounds(x_lims,y_lims)
        window_x=(x_lims[0],x_lims[0]+self.window_size[1])
        window_y=(y_lims[0],y_lims[0]+self.window_size[0])
        return window_x,window_y

    def get_data_range(self,x_lims,y_lims):
        map_len_x=np.ceil(x_lims[1]-x_lims[0])
        map_len_y=np.ceil(y_lims[1]-y_lims[0])
        return map_len_x,map_len_y

    def get_padded_bounds(self,x_lims,y_lims):

        map_len_x,map_len_y=self.get_data_range(x_lims,y_lims)

        padded_x_lims=(x_lims[0]- (map_len_x * self.padding/2), x_lims[1] + (map_len_x * self.padding/2))
        padded_y_lims=(y_lims[0]- (map_len_y * self.padding/2), y_lims[1] + (map_len_y * self.padding/2))

        return padded_x_lims,padded_y_lims

    def pad_data(self,x,y):
        
        x_lims,y_lims=self.get_data_bounds(x,y)
        map_len_x,map_len_y=self.get_data_range(x_lims,y_lims)

        translation_factor_x=map_len_x * self.padding/2
        translation_factor_y=map_len_y * self.padding/2

        x_new=x+translation_factor_x
        y_new=y+translation_factor_y

        return x_new,y_new

    def resize_input_data(self):
        
        #self.test_function(1,[1,2,3])
        input_array=self.data[[self.x_col,self.y_col]].values

        x_lims,y_lims=self.get_data_bounds(input_array[:,0],input_array[:,1])
        #print(x_lims)
        x_resize_factor=self.resize_param[0]/(x_lims[1]-x_lims[0])
        y_resize_factor=self.resize_param[1]/(y_lims[1]-y_lims[0])
        #print(x_resize_factor)
        #print(resized_data)
        x=input_array[:,0]*x_resize_factor
        y=input_array[:,1]*y_resize_factor
        #print(resized_data)
        return x,y


    def get_targets_from_window(self,x,y,window_x,window_y,target_vals):
        x_indices=np.where(np.logical_and(x>=window_x[0], x<=window_x[1]))
        y_indices=np.where(np.logical_and(y>=window_y[0], y<=window_y[1]))
        #print(y_indices)
        common_indices=set(x_indices[0]).intersection(set(y_indices[0]))
        #if len(common_indices)>0:
            #print(common_indices)
        target_vals_in_window=np.take(target_vals,list(common_indices))
        return target_vals_in_window

    def compute_density_in_window(self,targets_in_window):
        #total=len(targets_in_window)
        positives=targets_in_window.sum()
        return positives


    def get_density_map(self):
        # resize input data to the plot size
        x,y=self.resize_input_data()
        target=self.data[self.class_col]

        #print(x)

        # initialize output map
        output_map=self.init_output_map(x,y)

        print(output_map.shape)
        
        # pad data
        x_padded,y_padded=self.pad_data(x,y)

        window_x,window_y=self.init_window(x,y)

        for i in tqdm(reversed(list(range(output_map.shape[0])))):
            window_x,_=self.init_window(x,y)
            for j in list(range(output_map.shape[1])):
                t=self.get_targets_from_window(x_padded,y_padded,window_x,window_y,target)
                if len(t)==0:
                    output_map[i,j]=0
                else:
                    d=self.compute_density_in_window(t)
                    output_map[i,j]=d
                #print(window_x, ' ', window_y)
                window_x=(window_x[0]+self.stride,window_x[1]+self.stride)
            #print(window_y)
            window_y=(window_y[0]+self.stride,window_y[1]+self.stride)

        return output_map,np.column_stack([x_padded,y_padded])


    def plot_graph():
        return



if __name__=="__main__":
    print("you just ran density_map.py")