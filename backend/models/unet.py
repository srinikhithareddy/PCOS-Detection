"""
U-Net Architecture for Ovarian Follicle Segmentation
This file contains ONLY the U-Net model architecture - no weights, no inference
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional


class UNet:
    """U-Net architecture for binary segmentation"""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (512, 512),
        num_classes: int = 1,
        filters: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1
    ):
        """
        Initialize U-Net architecture parameters
        
        Args:
            input_size: Input image size (height, width)
            num_classes: Number of output classes (1 for binary segmentation)
            filters: Number of filters in first convolutional layer
            depth: Depth of the encoder (number of downsampling steps)
            dropout_rate: Dropout rate for regularization
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self) -> Model:
        """
        Build the U-Net model architecture
        
        Returns:
            Compiled Keras Model
        """
        inputs = layers.Input(shape=(*self.input_size, 3))
        
        # Encoder (contracting path)
        encoder_outputs = []
        x = inputs
        
        for i in range(self.depth):
            filters = self.filters * (2 ** i)
            x = self._conv_block(x, filters)
            encoder_outputs.append(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Bridge
        filters = self.filters * (2 ** self.depth)
        x = self._conv_block(x, filters)
        
        # Decoder (expanding path)
        for i in reversed(range(self.depth)):
            filters = self.filters * (2 ** i)
            x = layers.UpSampling2D(size=(2, 2))(x)
            x = layers.Concatenate()([x, encoder_outputs[i]])
            x = self._conv_block(x, filters)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes == 1:
            # Binary segmentation
            outputs = layers.Conv2D(
                1, 
                kernel_size=(1, 1), 
                activation='sigmoid',
                padding='same'
            )(x)
        else:
            # Multi-class segmentation
            outputs = layers.Conv2D(
                self.num_classes,
                kernel_size=(1, 1),
                activation='softmax',
                padding='same'
            )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='U-Net')
        
        self.model = model
        return model
    
    def _conv_block(self, x: layers.Layer, filters: int) -> layers.Layer:
        """
        Convolutional block with two convolutions and batch normalization
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor after convolutions
        """
        x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string
        
        Returns:
            Model summary string
        """
        if self.model is None:
            self.build_model()
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        loss: str = 'binary_crossentropy',
        metrics: Optional[list] = None
    ) -> Model:
        """
        Compile the model for training
        
        Args:
            optimizer: Optimizer name or optimizer instance
            loss: Loss function
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        if self.model is None:
            self.build_model()
        
        if metrics is None:
            metrics = ['accuracy']
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model


def create_unet_model(
    input_size: Tuple[int, int] = (512, 512),
    num_classes: int = 1,
    filters: int = 64,
    depth: int = 4,
    dropout_rate: float = 0.1
) -> Model:
    """
    Convenience function to create a U-Net model
    
    Args:
        input_size: Input image size (height, width)
        num_classes: Number of output classes
        filters: Number of filters in first layer
        depth: Depth of encoder
        dropout_rate: Dropout rate
        
    Returns:
        U-Net Keras Model
    """
    unet = UNet(input_size, num_classes, filters, depth, dropout_rate)
    return unet.build_model()
