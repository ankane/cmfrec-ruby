import pandas as pd
from cmfrec import CMF

ratings = pd.read_csv("test/support/ratings.csv")
ratings.columns = ["UserId", "ItemId", "Rating"]
user_info = pd.read_csv("test/support/user_info.csv")
user_info.rename(columns={"user_id": "UserId"}, inplace=True)
item_info = pd.read_csv("test/support/item_info.csv")
item_info.rename(columns={"item_id": "ItemId"}, inplace=True)

model = CMF(k=8, verbose=False)
model.fit(X=ratings, U=user_info, I=item_info)

print(list(model.predict(user=[3, 3], item=[2, 4])))
