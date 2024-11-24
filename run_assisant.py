from clustering_assistant import ClusteringAssistant
from mapper import Mapper

main_menu = '1. Cluster with DBSCAN\n2. Cluster with K-Means\n3. Map labels from sample to clustered data\n4. End program'
sub_menu = '1. Save clusters in separated csv files\n2. Create sample\n3. Display cluster centroids\n4. Display and save cluster sizes\n5. Back to main menu'
dbscan_menu = '1. Save clusters in separted csv files\n2. Create sample\n3. Display and save cluster sizes\n 4. Back to main menu'

end_running_main = False
while not end_running_main:
    print('[----MAIN MENU----]')
    print(main_menu)
    option = input('Option: ')

    if option == "1":
        input_file = input("Provide the name of your csv file with .csv ending: ")
        
        assistant = ClusteringAssistant(input_file)
        dbscan = assistant.dbscan_clustering()
        end_running = False
        while not end_running:
            print("[----SUB MENU----]")
            print(dbscan_menu)
            sub_option = input("Option: ")
            if sub_option == "1":
                assistant.save_clusters(dbscan)
            elif sub_option == "2":
                assistant.create_training_data(dbscan)
            elif sub_option == "3":
                assistant.display_cluster_sizes()
            elif sub_option == "4":
                end_running= True
            else:
                print("Invalid input")
    elif option == "2":
        input_file = input("Provide the name of your csv file with .csv ending: ")
        
        assistant = ClusteringAssistant(input_file)
        assistant.BestofK()
        n_clusters = int(input(" How many clusters do you want to get ?"))
        kmeans = assistant.kmeans_clustering(n_clusters)
        end_running = False
        while not end_running:
            print("[----SUB MENU----]")
            print(sub_menu)
            sub_option = input("Option: ")
            if sub_option == "1":
                assistant.save_clusters(kmeans)
            elif sub_option == "2":
                assistant.create_training_data(kmeans)
            elif sub_option == "3":
                assistant.display_cluster_centroids(kmeans)
            elif sub_option == "4":
                assistant.display_cluster_sizes()
            elif sub_option == "5":
                end_running= True
            else:
                print("Invalid input")
    elif option == "3":
        clustered_data = input("Please provide the name of your clustered file with .csv ending: ")
        decisions = input("Please provide the name of your labeled file with .csv ending:")
        map = Mapper(clustered_data, decisions)
        map.get_dataframe()

    elif option == "4":
        end_running_main = True
    else:
        print("Invalid input")

