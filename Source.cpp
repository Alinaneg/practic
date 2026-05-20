#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <map>
#include <iomanip>

using namespace std;

const vector<int> VALID_X = { 5, 6, 7, 8, 9 };
const int BITS = 3;
const double TARGET_F = 65.0;
const int TARGET_X = 5;

map<string, int> decode_map = {
    {"000", 5}, {"001", 6}, {"010", 7},
    {"011", 8}, {"100", 9}
};
map<int, string> encode_map = {
    {5, "000"}, {6, "001"}, {7, "010"},
    {8, "011"}, {9, "100"}
};

double f(int x) {
    return 3.0 * x + 2.0 * x * x;
}

double fitness(int x) {
    return 189.0 - f(x) + 1.0;
}

string fix_binary(string bits) {
    if (decode_map.count(bits)) return bits;
    if (bits == "101") return "100";
    if (bits == "110") return "011";
    return "100";
}

random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> dis(0.0, 1.0);
uniform_int_distribution<> pos_dis(0, BITS - 1);

string encode(int x) { return encode_map[x]; }
int decode(string bits) { return decode_map[bits]; }

vector<string> create_population_drobovik(int pop_size) {
    vector<string> pop;
    uniform_int_distribution<> x_dis(0, VALID_X.size() - 1);
    for (int i = 0; i < pop_size; i++) {
        pop.push_back(encode(VALID_X[x_dis(rng)]));
    }
    return pop;
}

vector<string> create_population_fokusirovka(int pop_size) {
    vector<string> pop;
    vector<int> weights = { 30, 25, 20, 15, 10 };
    discrete_distribution<> dist(weights.begin(), weights.end());
    for (int i = 0; i < pop_size; i++) {
        pop.push_back(encode(VALID_X[dist(rng)]));
    }
    return pop;
}

vector<string> selection_random(const vector<string>& pop) {
    int size = pop.size();
    vector<string> parents;
    uniform_int_distribution<> idx_dis(0, size - 1);
    for (int i = 0; i < size; i++) {
        parents.push_back(pop[idx_dis(rng)]);
    }
    return parents;
}

vector<string> selection_elite(const vector<string>& pop) {
    int size = pop.size();
    vector<pair<double, string>> scored;
    for (const auto& ind : pop) {
        scored.push_back({ fitness(decode(ind)), ind });
    }
    sort(scored.begin(), scored.end(),
        [](const pair<double, string>& a, const pair<double, string>& b) {
            return a.first > b.first;
        });
    vector<string> parents;
    for (int i = 0; i < size; i++) {
        parents.push_back(scored[i % (size / 2)].second);
    }
    return parents;
}

pair<string, string> crossover_A(const string& p1, const string& p2) {
    int point = uniform_int_distribution<>(1, BITS - 1)(rng);
    return { fix_binary(p1.substr(0, point) + p2.substr(point)),
            fix_binary(p2.substr(0, point) + p1.substr(point)) };
}

pair<string, string> crossover_B(const string& p1, const string& p2) {
    int point1 = uniform_int_distribution<>(1, BITS - 2)(rng);
    int point2 = uniform_int_distribution<>(point1 + 1, BITS - 1)(rng);
    string c1 = p1.substr(0, point1) + p2.substr(point1, point2 - point1) + p1.substr(point2);
    string c2 = p2.substr(0, point1) + p1.substr(point1, point2 - point1) + p2.substr(point2);
    return { fix_binary(c1), fix_binary(c2) };
}

pair<string, string> crossover_E(const string& p1, const string& p2) {
    return crossover_B(p1, p2);
}

pair<string, string> crossover_L(const string& p1, const string& p2) {
    string c1 = p1.substr(0, 1) + p2.substr(1, 1) + p1.substr(2);
    string c2 = p2.substr(0, 1) + p1.substr(1, 1) + p2.substr(2);
    return { fix_binary(c1), fix_binary(c2) };
}

string mutation_A(const string& ind) {
    string mutated = ind;
    if (dis(rng) < 0.2) {
        int pos = pos_dis(rng);
        mutated[pos] = (mutated[pos] == '0') ? '1' : '0';
    }
    return fix_binary(mutated);
}

string mutation_I(const string& ind) {
    string mutated = ind;
    if (dis(rng) < 0.2) {
        int pos1 = pos_dis(rng);
        int pos2 = pos_dis(rng);
        swap(mutated[pos1], mutated[pos2]);
    }
    return fix_binary(mutated);
}

vector<string> elite_selection(const vector<string>& old_pop, const vector<string>& new_pop) {
    vector<pair<double, string>> all;
    for (const auto& ind : old_pop) all.push_back({ fitness(decode(ind)), ind });
    for (const auto& ind : new_pop) all.push_back({ fitness(decode(ind)), ind });
    sort(all.begin(), all.end(),
        [](const pair<double, string>& a, const pair<double, string>& b) {
            return a.first > b.first;
        });
    vector<string> result;
    for (size_t i = 0; i < old_pop.size(); i++) result.push_back(all[i].second);
    return result;
}

pair<vector<double>, int> run_evolution(int pop_size, int generations,
    double pcross, double pmut,
    int pop_strategy, int sel_type,
    int cross_type, int mut_type) {
    vector<string> population;
    if (pop_strategy == 0) population = create_population_drobovik(pop_size);
    else population = create_population_fokusirovka(pop_size);

    vector<double> avg_history;
    int optimum_gen = -1;

    for (int gen = 0; gen < generations; gen++) {
        vector<string> parents = (sel_type == 0) ? selection_random(population) : selection_elite(population);

        vector<string> offspring;
        for (size_t i = 0; i < parents.size(); i += 2) {
            if (dis(rng) < pcross) {
                pair<string, string> children;
                switch (cross_type) {
                case 0: children = crossover_A(parents[i], parents[i + 1]); break;
                case 1: children = crossover_B(parents[i], parents[i + 1]); break;
                case 2: children = crossover_E(parents[i], parents[i + 1]); break;
                case 3: children = crossover_L(parents[i], parents[i + 1]); break;
                default: children = crossover_A(parents[i], parents[i + 1]); break;
                }
                offspring.push_back(children.first);
                offspring.push_back(children.second);
            }
            else {
                offspring.push_back(parents[i]);
                offspring.push_back(parents[i + 1]);
            }
        }

        for (auto& ind : offspring) {
            ind = (mut_type == 0) ? mutation_A(ind) : mutation_I(ind);
        }

        population = elite_selection(population, offspring);

        double sum_f = 0.0;
        for (const auto& ind : population) {
            sum_f += f(decode(ind));
        }
        double avg_f = sum_f / population.size();
        avg_history.push_back(avg_f);

        if (optimum_gen == -1 && fabs(avg_f - 65.0) < 1e-6) {
            optimum_gen = gen;
        }
    }
    return { avg_history, optimum_gen };
}

void save_subplot_data(const string& filename, const vector<double>& history) {
    ofstream file(filename);
    for (size_t i = 0; i < history.size(); i++) {
        file << i << " " << history[i] << "\n";
    }
    file.close();
}

void save_optimum_point(const string& filename, int gen, double value) {
    ofstream file(filename);
    file << gen << " " << value << "\n";
    file.close();
}

void run_and_save(int pop_size, int generations, double pcross, double pmut, const string& suffix) {
    string base = "pop" + to_string(pop_size) + "_gen" + to_string(generations) +
        "_pc" + to_string((int)(pcross * 100)) + "_pm" + to_string((int)(pmut * 100)) + suffix;

    string groups[4] = { "B_A", "B_C", "C_A", "C_C" };
    string group_titles[4] = {
        "Drobovik + Sluchaynaya selektsiya",
        "Drobovik + Elitnaya selektsiya",
        "Fokusirovka + Sluchaynaya selektsiya",
        "Fokusirovka + Elitnaya selektsiya"
    };

    string combo_names[8] = {
        "One-point crossingover + Simple mutation",
        "One-point crossingover + Transposition mutation",
        "Two-point crossingover + Simple mutation",
        "Two-point crossingover + Transposition mutation",
        "Ordering crossingover + Simple mutation",
        "Ordering crossingover + Transposition mutation",
        "Fibonacci crossingover + Simple mutation",
        "Fibonacci crossingover + Transposition mutation"
    };

    for (int ps = 0; ps < 2; ps++) {
        for (int sel = 0; sel < 2; sel++) {
            int group_idx = ps * 2 + sel;

            vector<vector<double>> histories(8);
            vector<int> optimum_gens(8, -1);

            for (int cross = 0; cross < 4; cross++) {
                for (int mut = 0; mut < 2; mut++) {
                    int idx = cross * 2 + mut;
                    auto result = run_evolution(pop_size, generations, pcross, pmut,
                        ps, sel, cross, mut);
                    histories[idx] = result.first;
                    optimum_gens[idx] = result.second;
                }
            }

            for (int idx = 0; idx < 8; idx++) {
                string data_file = base + "_" + groups[group_idx] + "_" + to_string(idx) + ".txt";
                save_subplot_data(data_file, histories[idx]);

                if (optimum_gens[idx] != -1) {
                    string point_file = base + "_" + groups[group_idx] + "_" + to_string(idx) + "_opt.txt";
                    save_optimum_point(point_file, optimum_gens[idx], 65.0);
                }
            }

            string plot_file = base + "_plot_" + groups[group_idx] + ".gp";
            ofstream gp(plot_file);

            gp << "set terminal png size 1200,1600\n";
            gp << "set output \"" << base << "_graph_" << groups[group_idx] << ".png\"\n";
            gp << "set multiplot layout 4,2 title \"" << group_titles[group_idx] << "\"\n";
            gp << "set xlabel 'Generation'\n";
            gp << "set ylabel 'Average f(x)'\n";
            gp << "set xrange [0:5]\n";
            gp << "set yrange [60:100]\n";
            gp << "set grid\n";
            gp << "set key at screen 0.95, 0.95\n";

            for (int idx = 0; idx < 8; idx++) {
                gp << "set title '" << combo_names[idx] << "'\n";
                gp << "plot \"" << base << "_" << groups[group_idx] << "_" << idx << ".txt\" using 1:2 with lines lc rgb 'blue' notitle, \\\n";
                gp << "     \"" << base << "_" << groups[group_idx] << "_" << idx << ".txt\" using 1:2 with points pt 7 lc rgb 'blue' notitle";

                if (optimum_gens[idx] != -1) {
                    gp << ", \\\n     \"" << base << "_" << groups[group_idx] << "_" << idx << "_opt.txt\" using 1:2 with points pt 7 ps 2 lc rgb 'red' title 'Optimum'";
                }
                gp << "\n\n";
            }

            gp << "unset multiplot\n";
            gp.close();

            string cmd = "gnuplot " + plot_file;
            system(cmd.c_str());

            cout << "График сохранён: " << base << "_graph_" << groups[group_idx] << ".png\n";
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    cout << "Запуск с параметрами по умолчанию: размер популяции = 10, число поколений = 50, вероятность кроссинговера = 0.7, вероятность мутации = 0.2\n";
    run_and_save(10, 50, 0.7, 0.2, "_default");
    cout << "\nХотите ввести свои параметры? (y/n): ";
    char choice;
    cin >> choice;

    if (choice == 'y' || choice == 'Y') {
        int pop_size, generations;
        double pcross, pmut;

        cout << "Введите размер популяции: ";
        cin >> pop_size;
        if (pop_size % 2 != 0) pop_size++;

        cout << "Введите число поколений (минимум 50): ";
        cin >> generations;
        if (generations < 50) generations = 50;

        cout << "Введите вероятность кроссинговера (0-1): ";
        cin >> pcross;

        cout << "Введите вероятность мутации (0-1): ";
        cin >> pmut;

        cout << "\nЗапуск с пользовательскими параметрами: размер популяции = " << pop_size
            << ", число поколений = " << generations
            << ", вероятность кроссинговера = " << pcross
            << ", вероятность мутации = " << pmut << "\n";

        run_and_save(pop_size, generations, pcross, pmut, "_user");
    }
    else {
        cout << "Работа программы завершена.\n";
    }

    return 0;
}