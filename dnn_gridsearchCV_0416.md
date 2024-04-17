`def build_model(optimizer):
    # Define and compile the model
    model = keras.Sequential([
        Dense(45, input_dim=90, activation='relu'),
        Dropout(0.2),
        Dense(45, activation='relu'),
        Dropout(0.2),
        Dense(45, activation='relu'),
        Dropout(0.2),
        Dense(10),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
skfold = KFold(random_state=30,
           n_splits=5,
           shuffle=True
          )
model = KerasClassifier(build_fn = build_model)
parameters = {'batch_size': [1000,
                             10000],
              'epochs': [10,
                         20,
                         40,
                         60,
                         80,
                         100],
              'optimizer': ['adam', 
                            'SGD', 
                            'rmsprop'
                            ]}
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = skfold)
# early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.01)
grid_search.fit(x_train, y_train)
print("최고의 파라미터 :", grid_search.best_params_)
print("최고 평균 정확도 : {}".format(grid_search.best_score_))`