def prepare_model_for_kbit_training(model):
    for _, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
    model.enable_input_require_grads()