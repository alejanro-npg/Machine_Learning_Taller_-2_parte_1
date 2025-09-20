from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from prefect import flow, task

@task(name="load_datasets")
def load_datasets():
    # dataset cancer sklearn
    d_cancer = datasets.load_breast_cancer()

    # iris desde csv
    d_iris = pd.read_csv("iris.csv")

    # diabetes desde csv
    nombre_columnas = ['Embarazos', 'Glucosa', 'Presion Arterial', 'Grosor Piel', 
                       'Insulina', 'Indice Masa Corporal', 'Funcion de Pedigree', 
                       'Edad', 'Etiqueta']
    d_diabetes = pd.read_csv("diabetes.csv", header=1, names=nombre_columnas)

    return d_cancer, d_iris, d_diabetes

@task(name="split_datasets")
def split_datasets(d_cancer, d_iris, d_diabetes):
    datasets_splits = {}

    # cancer
    X_cancer, y_cancer = d_cancer.data, d_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer
    )
    datasets_splits["cancer"] = (X_train, X_test, y_train, y_test, d_cancer.feature_names, d_cancer.target_names)

    # iris
    X_iris = d_iris.drop(columns=["Species"])
    y_iris = d_iris["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )
    datasets_splits["iris"] = (X_train, X_test, y_train, y_test, X_iris.columns, y_iris.unique())

    # diabetes
    X_diabetes = d_diabetes.drop(columns=["Etiqueta"])
    y_diabetes = d_diabetes["Etiqueta"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_diabetes, y_diabetes, test_size=0.3, random_state=42, stratify=y_diabetes
    )
    datasets_splits["diabetes"] = (X_train, X_test, y_train, y_test, X_diabetes.columns, ["0", "1"])

    return datasets_splits

@task(name="modelos")
def modelos(splits):
    modelos = {
        "KNN_k3": KNeighborsClassifier(n_neighbors=3),
        "KNN_k7": KNeighborsClassifier(n_neighbors=7),
        "SVM_linear": SVC(kernel="linear", probability=True, random_state=42),
        "SVM_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    resultados = []
    detalles = {}
    os.makedirs("resultados", exist_ok=True)

    for dataset_name, (X_train, X_test, y_train, y_test, feature_names, target_names) in splits.items():
        for model_name, model in modelos.items():
            inicio = time.time()
            model.fit(X_train, y_train)
            tiempo_ent = time.time() - inicio

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

            resultados.append({
                "Dataset": dataset_name,
                "Modelo": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "Tiempo": tiempo_ent
            })

            detalles[(dataset_name, model_name)] = {
                "y_test": y_test,
                "y_pred": y_pred,
                "tiempo": tiempo_ent,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

            # matriz de confusion
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"matriz de confusion - {dataset_name} - {model_name}")
            plt.xlabel("prediccion")
            plt.ylabel("real")
            plt.savefig(f"resultados/{dataset_name}_{model_name}_confusion.png")
            plt.close()

            # curva roc si es binario
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"auc = {auc:.2f}")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.xlabel("false positive rate")
                    plt.ylabel("true positive rate")
                    plt.title(f"roc - {dataset_name} - {model_name}")
                    plt.legend(loc="lower right")
                    plt.savefig(f"resultados/{dataset_name}_{model_name}_roc.png")
                    plt.close()

            # arbol
            if model_name == "DecisionTree":
                plt.figure(figsize=(12, 8))
                plot_tree(model, 
                          filled=True, 
                          feature_names=feature_names, 
                          class_names=target_names, 
                          rounded=True, 
                          fontsize=8)
                plt.title(f"arbol de decision - {dataset_name}")
                plt.savefig(f"resultados/{dataset_name}_DecisionTree.png")
                plt.close()

    return pd.DataFrame(resultados), detalles


@task(name="workshop_questions")
def workshop_questions(splits, resultados, detalles):
    print("\n--- workshop questions ---")
    df = resultados

    # 1. efecto de no escalar
    dataset_name = "cancer"
    X_train, X_test, y_train, y_test, _, _ = splits[dataset_name]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    svm = SVC(probability=True, random_state=42)

    knn.fit(X_train, y_train)
    acc_knn_no = accuracy_score(y_test, knn.predict(X_test))
    knn.fit(X_train_scaled, y_train)
    acc_knn_si = accuracy_score(y_test, knn.predict(X_test_scaled))

    svm.fit(X_train, y_train)
    acc_svm_no = accuracy_score(y_test, svm.predict(X_test))
    svm.fit(X_train_scaled, y_train)
    acc_svm_si = accuracy_score(y_test, svm.predict(X_test_scaled))

    print(f"\n1. Efecto de no escalar (ejemplo en cancer):")
    print(f"kNN sin: {acc_knn_no:.4f}, con: {acc_knn_si:.4f}")
    print(f"SVM sin: {acc_svm_no:.4f}, con: {acc_svm_si:.4f}")
    print("Se evidencia que al escalar los datos, tanto kNN como SVM mejoran su accuracy.")

    # 2. mas rapido
    fastest = df.loc[df["Tiempo"].idxmin()]
    print(f"\n2. Modelo más rápido: {fastest['Modelo']} en {fastest['Dataset']} ({fastest['Tiempo']:.4f}s)")

    # 3. rf vs otros
    print("\n3. Comparación de Random Forest vs otros:")
    for dataset in df["Dataset"].unique():
        subset = df[df["Dataset"] == dataset]
        rf_acc = subset[subset["Modelo"] == "RandomForest"]["Accuracy"].values[0]
        mejor = subset.loc[subset["Accuracy"].idxmax()]
        print(f"- En {dataset}, RandomForest={rf_acc:.4f} vs mejor modelo ({mejor['Modelo']})={mejor['Accuracy']:.4f}")

    # 4. hiperparametros
    print("\n4. Efecto de hiperparámetros:")
    knn_acc = df[df["Modelo"].str.contains("KNN")].groupby("Modelo")["Accuracy"].mean()
    svm_acc = df[df["Modelo"].str.contains("SVM")].groupby("Modelo")["Accuracy"].mean()
    print(f"kNN k=3 -> {knn_acc['KNN_k3']:.4f}, k=7 -> {knn_acc['KNN_k7']:.4f}")
    print(f"SVM linear -> {svm_acc['SVM_linear']:.4f}, rbf -> {svm_acc['SVM_rbf']:.4f}")
    print("Se evidencia el impacto de cambiar hiperparámetros: en k=3 y el kernel linear de SVM, se mejora el accuracy promedio.")

    # 5. mejor modelo en cada dataset
    print("\n5. desempeño por clase del mejor modelo en cada dataset:")
    for dataset in df["Dataset"].unique():
        subset = df[df["Dataset"] == dataset]
        mejor = subset.loc[subset["Accuracy"].idxmax()]
        rep = detalles[(mejor["Dataset"], mejor["Modelo"])]["classification_report"]

        print(f"\nDataset: {dataset}")
        print(f"Mejor modelo: {mejor['Modelo']} con accuracy {mejor['Accuracy']:.4f}")
        print("Porcentaje de aciertos (recall) por clase:")
        for clase, metrics in rep.items():
            if clase not in ["accuracy", "macro avg", "weighted avg"]:
                print(f"  Clase {clase}: {metrics['recall']*100:.2f}%")


@flow(name="Taller2 Workflow")
def main():
    d_cancer, d_iris, d_diabetes = load_datasets()
    splits = split_datasets(d_cancer, d_iris, d_diabetes)
    resultados, detalles = modelos(splits)

    print("\nResultados generales:")
    print(resultados)

    print("\nMejores modelos por dataset:")
    for dataset in resultados["Dataset"].unique():
        subset = resultados[resultados["Dataset"] == dataset]
        mejor = subset.loc[subset["Accuracy"].idxmax()]
        print(f"{dataset}: {mejor['Modelo']} con accuracy {mejor['Accuracy']:.4f}")

    workshop_questions(splits, resultados, detalles)


if __name__ == "__main__":
    main()
