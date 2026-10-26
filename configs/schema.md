# Experiment config schema

One YAML file per run. The file has exactly these keys:

```text
run_id
git_commit
seed
dataset_name          # faceforensics++
dataset_version       # the release identifier Jas records at download
split_id              # official-720-140-140
train_video_list      # path to a file of video ids
quality               # raw | c23 | c40
frame_sample          # the rule used to pick frames
architecture          # xception-face | xception-full | steg-svm | mesonet | ...
class_weight          # the imbalance weights
optimizer             # paper values for this architecture
max_epochs
early_stopping
held_out_methods      # empty for joint training
perturbation          # none | fgsm | pgd plus the budget
notes
```

A key may be null until the run fills it. A key outside this list is not part of the schema.
