#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <map>
#include <math.h>
#include <set>
using namespace std;
// fp-tree实现
typedef unsigned long ul;
struct Node{
    string s;
    int count;
    vector<Node> child;
    map<string,int>child_index; // children index
    pair<string,int> pair_path;
    Node(string s,int c):s(s),count(c){
    }
};
void dfs(Node root,int dep,vector<string> tmp,vector<Node> &origin, map<string,int> &origin_index);
void buildHeaderTable(vector<vector<string>> data,vector<pair<string,int>> suffix,vector<vector<int>> data_value);
int min_sup = 771;
vector<string> split(string pattern,string str){
    vector<string>result;
    str += pattern;
    string sub_str = "";
    ul size = str.size();
    for(ul i = 0;i < size ;i ++){
        unsigned long  pos = str.find(pattern,i);
        if(pos < size){
            sub_str = str.substr(i,pos - i);
            result.push_back(sub_str);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}
string vector2string(vector<string> path){
    string result = "";
    for(string tmp : path){
        result += tmp;
        result += ";";
    }
    result = result.substr(0,result.size() -1);
    return result;
}
struct PairCompare {
    bool operator() (const pair<string,int> &pair1, const pair<string,int> &pair2){
        return pair1.second > pair2.second;
    }
};
// 2^cnt - 1
int getSum(int cnt){
    int ans = 1;
    for(int i = 0;i<cnt; i++){
        ans *=2;
    }
    return ans - 1;
}
bool isSinglePath(Node root,vector<pair<string,int>> &path,int dep){
    if(root.child.size() > 1){
        return false;
    }
    if(root.child.size() == 0){
        if(dep != 0) {
            path.push_back(make_pair(root.s, root.count));
        }
        return true;
    }
    else{
        if(dep != 0) {
            path.push_back(make_pair(root.s, root.count));
        }
        return isSinglePath(root.child[0],path,dep + 1);
    }
}

void combination(vector<pair<string,int>> path, vector<pair<string,int>> suffix){
    int size = path.size();
    int sum = getSum(size);
    for(int i = 0;i <= sum ;i++){
        int mi = suffix[suffix.size() - 1].second;
        for(int j=0;j<path.size();j++){
            if((1 << j) & i ){
                cout << path[j].first<<";";
                mi = min(mi,path[j].second);
            }
        }
        for(pair<string,int> tmp:suffix){
            cout<<tmp.first<<"; ";
        }
        cout <<mi<< endl;
    }
}

// map
void mapGenerate(vector<vector<string>> data,map<string,int> &mp,vector<vector<int>> data_value){
    mp.clear();

    for(int i=0;i<data.size();i++){
        for(int j=0;j<data[i].size();j++){
            mp[data[i][j]] += data_value[i][j];
            //cout<<data[i][j]<<";";
        }
        //cout<<endl;
    }
}
void create_tree_by_one_line(vector<string> data_i, Node &root_tmp,int count = 1){
    Node *root = &root_tmp;
    for(int i = 0;i<data_i.size();i++){
        //cout<< data_i[i]<<" ";
        if(root->child_index.find(data_i[i]) == root->child_index.end()){ // doesn't have the children
            root->child_index.insert(make_pair(data_i[i],root->child_index.size()));
            root->child.push_back(Node(data_i[i],count));
            root = &root->child[root->child.size()-1];
        }else{
            root->child[root->child_index[data_i[i]]].count += count;
            root = &root->child[root->child_index[data_i[i]]];
        }
    }
}
void build_tree(vector<vector<string>> data, vector<Node> origin, map<string,int> origin_index,vector<pair<string,int> >suffix,vector<vector<int>> data_value){

    Node root = Node("null", -1);
    for (int i = 0; i < data.size(); i++) {
        create_tree_by_one_line(data[i], root, data_value[i][0]);
    }
    vector<string> tmp;
    vector<pair<string, int>> path;
    if (isSinglePath(root, path, 0)) {
        // accroding to suffix,generate frequent pattern,
        combination(path, suffix);

    } else {
        dfs(root, 0, tmp, origin, origin_index);// generate path
        //generate origin[0] frequent pattern
        cout << origin[0].s <<"; ";
        for (pair<string,int> tmp : suffix) {
            cout << tmp.first << "; ";
        }
        cout << origin[0].count << endl;

        //vector<vector<int>> data_value;
        for (int i = 1; i < origin.size(); i++) {
            data.clear();
            data_value.clear();
            for (int j = 0; j < origin[i].child.size(); j++) {
                vector<string> data_i = split(";", origin[i].child[j].pair_path.first);
                vector<int> data_j;
                data_i.erase(data_i.begin());
                for (int k = 0; k < data_i.size(); k++) {
                    data_j.push_back(origin[i].child[j].pair_path.second);
                }
                if(data_i.size() > 0) {
                    data.push_back(data_i);
                    data_value.push_back(data_j);
                }
            }
            suffix.push_back(make_pair(origin[i].s,origin[i].count)); // add origin.s to suffix
            buildHeaderTable(data, suffix, data_value);
            suffix.pop_back();
        }
    }

}
void buildHeaderTable(vector<vector<string>> data,vector<pair<string,int>> suffix,vector<vector<int>> data_value){
    vector<Node> origin;
    map<string,int> origin_index;
    map<string,int> mp,mp_tmp; // for each line should sort we can use map as a tool
    mapGenerate(data,mp,data_value);
    map<string,int>::iterator mp_iterator = mp.begin();
    set<string> filter_element; // for the element suitable for condition
    for (; mp_iterator != mp.end();) {
        if (mp_iterator->second < min_sup) {
            //cout << "erase element" << mp_iterator->first << "  " << mp_iterator->second << " --\n";
            mp.erase(mp_iterator++);   //recommand
        } else {
            //generate frequent pattern
            mp_iterator++;
        }
    }
    vector<pair<string, int> > map2vector(mp.begin(), mp.end());
    sort(map2vector.begin(), map2vector.end(), PairCompare());
    // add to the origin vector
    for (pair<string, int> tmp :map2vector) {
        origin.push_back(Node(tmp.first, tmp.second));
        origin_index.insert(make_pair(tmp.first, origin_index.size())); //create index for origin
        filter_element.insert(tmp.first);
    }
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size();) {
            if (filter_element.find(data[i][j]) == filter_element.end()) // not found the element,remove
            {
                data[i].erase(data[i].begin() + j);
            } else {
                j++;
            }
        }
        // sort the data
        mp_tmp.clear(); // clear
        for (int j = 0; j < data[i].size(); j++) {
            mp_tmp.insert(make_pair(data[i][j], mp[data[i][j]]));
        }
        vector<pair<string, int> > map2vector_1(mp_tmp.begin(), mp_tmp.end());
        sort(map2vector_1.begin(), map2vector_1.end(), PairCompare());
        data[i].clear();
        for (int j = 0; j < map2vector_1.size(); j++) {
            data[i].push_back(map2vector_1[j].first);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        data_value[i].erase(unique(data_value[i].begin(), data_value[i].end()), data_value[i].end());
    }
    build_tree(data, origin, origin_index, suffix, data_value); //

}
void dfs(Node root,int dep,vector<string> tmp,vector<Node> &origin, map<string,int> &origin_index){
    //cout<< "string: "<< root.s << "count: "<<root.count<< "size of children: "<<root.child.size() <<" dep : "<< dep<<endl;
    string prefix = vector2string(tmp);
    root.pair_path = make_pair(prefix,root.count);
    origin[origin_index[root.s]].child.push_back(root); // add to origin
    //cout<< " prefix_path : "<<prefix << "  count: "<<root.count  << "  string: "<<root.s<<endl;
    if(root.child.size() == 0){
        return;
    }
    for(int i = 0;i<root.child.size();i++){
        tmp.push_back(root.s);
        dfs(root.child[i],dep + 1,tmp,origin,origin_index);
        tmp.pop_back();
    }
}
void view_origin(vector<Node> origin){
    for(int i=1;i<origin.size();i++){
        for(int j= 0 ;j<origin[i].child.size();j++){
            cout<< "prefix_path : "<< origin[i].child[j].pair_path.first << "count : "<< origin[i].child[j].pair_path.second<<endl;
        }
    }
}
int main() {
    freopen("../data/data.txt","r",stdin);
    //freopen("data.out","w",stdout);
    string lines;
    vector<vector<string>> data;
    while(getline(cin,lines)){
        vector<string> vector_tmp = split(";",lines);
        vector_tmp.erase(unique(vector_tmp.begin(),vector_tmp.end()),vector_tmp.end());
        data.push_back(vector_tmp);
    }
    vector<pair<string,int>> suffix;
    vector<vector<int>> data_value;
    for(int i=0;i<data.size();i++){
        vector<int>data_i;
        for(int j=0;j<data[i].size();j++){
            data_i.push_back(1);
        }
        data_value.push_back(data_i);
    }
    buildHeaderTable(data,suffix,data_value);
    //view_origin();
    return 0;
}