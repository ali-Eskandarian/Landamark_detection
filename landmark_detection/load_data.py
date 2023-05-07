
class Load_data():
    def __init__(self, labels_path):
        self.labels_path = labels_path

    def parse_file(self):
        
        data = {"Name" : [], "Jaw, Nose and Euler Angles" : []}
        with open(self.labels_path, 'r') as list_of_labels:
            list_of_labels = list_of_labels.read().split("\n")
            for labels in list_of_labels:
                sub_labels = labels.split(" ")
                label_name = sub_labels[0].split("\\")[-1]
                njea = [float(point) for point in sub_labels[109:111] + sub_labels[33:35] + sub_labels[203:206]]
                data["Name"].append(label_name)
                data["Jaw, Nose and Euler Angles"].append(njea)
                yield label_name, njea
    
    def __iter__():
        pass

                

if __name__=="__main__":
    load = Load_data()
    for data in load.read_data("test_data/list.txt"):
        print(data)