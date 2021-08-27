from src.utils.CheckpointUtil import print_checkpoint_variables, rename_checkpoint_variables

str1 = 'config/model.ckpt-4788'
str2 = 'result/topic/model.ckpt-3000'
rename_checkpoint_variables('config/model.ckpt-4988', 'config/model.ckpt-5088')
print_checkpoint_variables('config/model.ckpt-5088')
