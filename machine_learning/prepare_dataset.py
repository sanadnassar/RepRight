# Aref
import pandas as pd
angles = pd.read_csv("angles.csv")
labels = pd.read_csv("labels.csv")
# step 1 merge both files
df = pd.merge(angles,labels, on='pose_id')

print("after merge:", df.shape)
print(df.head(3))
#step 2 only rows with squats down and up
squat_mask = df['pose'].isin(['squats_down', 'squats_up'])

df_squats = df[squat_mask]

print("Squat rows only:", df_squats.shape)
print(df_squats['pose'].value_counts())
# assign squats down = good (full depth) squats up = bad (no range of motion)
df_squats = df_squats.copy()
df_squats['label'] = df_squats['pose'].map({
    'squats_down': 'good',
    'squats_up': 'bad',
})


# step 5 drop columns no needed for now
columns_to_drop = [
    'pose_id',
    'pose',
    'right_wrist_right_elbow_right_shoulder',
    'left_wrist_left_elbow_left_shoulder'
]
df_final = df_squats.drop(columns=columns_to_drop)

print("Final dataset shape:", df_final.shape)
print(df_final['label'].value_counts())
print(df_final.head(3))

# step 6 save it into squats_dataset file
df_final.to_csv("squats_dataset.csv", index=False)
print("\nSaved squats_dataset.csv")