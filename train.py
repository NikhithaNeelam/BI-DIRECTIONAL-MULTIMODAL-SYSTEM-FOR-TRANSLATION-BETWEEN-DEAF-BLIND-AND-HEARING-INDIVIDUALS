"""
train.py — COMPLETE WORKING VERSION
Run: python train.py --data datasets/landmarks_isl.npz
     python train.py --data datasets/landmarks_all.npz
Outputs: model/isl_model.tflite, model/class_labels.json,
         model/feature_mean.npy, model/feature_std.npy
"""
import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(512), layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(0.4),
        layers.Dense(256), layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(0.3),
        layers.Dense(128), layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(0.2),
        layers.Dense(64),  layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax"),
    ], name="SignLangNet")
    return model

def train(data_path, epochs=50, batch_size=64, lr=1e-3):
    print(f"\nLoading: {data_path}")
    data    = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]; y_train = data["y_train"]
    X_test  = data["X_test"];  y_test  = data["y_test"]
    classes = list(data["classes"])
    n_cls   = len(classes); in_dim = X_train.shape[1]
    print(f"  Classes({n_cls}): {classes}")
    print(f"  Train:{len(X_train)}  Test:{len(X_test)}  Input dim:{in_dim}")

    # Normalise
    mean = X_train.mean(axis=0); std = X_train.std(axis=0) + 1e-8
    X_train = (X_train-mean)/std; X_test = (X_test-mean)/std
    os.makedirs("model", exist_ok=True)
    np.save("model/feature_mean.npy", mean)
    np.save("model/feature_std.npy",  std)

    y_tr_oh = keras.utils.to_categorical(y_train, n_cls)
    y_te_oh = keras.utils.to_categorical(y_test,  n_cls)

    model = build_model(in_dim, n_cls)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=12,
                                      restore_best_weights=True,verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,
                                          patience=6,verbose=1),
        keras.callbacks.ModelCheckpoint("model/best_model.keras",
                                        monitor="val_accuracy",save_best_only=True),
    ]
    history = model.fit(X_train, y_tr_oh, validation_data=(X_test,y_te_oh),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    loss,acc = model.evaluate(X_test, y_te_oh, verbose=0)
    print(f"\nTest accuracy: {acc*100:.2f}%")
    y_pred = np.argmax(model.predict(X_test,verbose=0),axis=1)
    print(classification_report(y_test, y_pred, target_names=classes))

    # Plot
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
    ax1.plot(history.history["accuracy"],label="Train"); ax1.plot(history.history["val_accuracy"],label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history["loss"],label="Train"); ax2.plot(history.history["val_loss"],label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig("training_history.png",dpi=120)
    print("  Saved training_history.png")

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tpath = "model/isl_model.tflite"
    with open(tpath,"wb") as f: f.write(tflite_model)
    print(f"  Saved {tpath} ({os.path.getsize(tpath)//1024} KB)")

    # Update class_labels.json
    with open("model/class_labels.json","w") as f:
        json.dump({"classes":classes,"num_classes":n_cls,
                   "test_accuracy":round(float(acc),4),"input_dim":in_dim},f,indent=2)

    # Quick TFLite test
    interp = tf.lite.Interpreter(model_path=tpath)
    interp.allocate_tensors()
    inp=interp.get_input_details(); outp=interp.get_output_details()
    sample=X_test[0:1].astype(np.float32)
    interp.set_tensor(inp[0]["index"],sample); interp.invoke()
    probs=interp.get_tensor(outp[0]["index"])[0]
    print(f"\n  Sample: predicted={classes[np.argmax(probs)]} actual={classes[y_test[0]]} conf={probs.max()*100:.1f}%")
    print("\nTraining complete!")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="datasets/landmarks_isl.npz")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch",type=int,default=64)
    parser.add_argument("--lr",type=float,default=1e-3)
    args=parser.parse_args()
    train(args.data,args.epochs,args.batch,args.lr)
