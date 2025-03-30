import random

# 적합도 함수: 충돌하지 않는 퀸 쌍의 수 계산
def fitness(solution):
    """
    solution: 퀸의 위치를 나타내는 리스트 (인덱스=열, 값=행)
    반환값: 28 - 충돌하는 쌍의 수 (적합도가 높을수록 좋은 해)
    """
    conflicts = 0
    # 모든 퀸 쌍에 대해 충돌 확인
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            # 같은 행에 있는 경우
            if solution[i] == solution[j]:
                conflicts += 1
            # 대각선에 있는 경우
            if abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1
    # 최대 적합도는 28 (모든 쌍이 충돌하지 않음)
    return 28 - conflicts

# 초기 개체군 생성
def create_population(pop_size):
    """
    pop_size: 개체군 크기
    반환값: 랜덤으로 생성된 해(solution) 리스트
    """
    population = []
    for _ in range(pop_size):
        # 0~7 사이의 값을 무작위로 배치 (행 위치)
        solution = random.sample(range(8), 8)  # 중복되지 않는 0~7의 순열
        population.append(solution)
    return population

# 선택: 룰렛 휠 방식으로 부모 선택
def select_parents(population, fitness_values):
    """
    population: 현재 개체군
    fitness_values: 각 해의 적합도 리스트
    반환값: 선택된 두 부모
    """
    total_fitness = sum(fitness_values)
    if total_fitness == 0:  # 적합도가 모두 0인 경우 방지
        total_fitness = 1
    pick1 = random.uniform(0, total_fitness)
    pick2 = random.uniform(0, total_fitness)

    current = 0
    parent1 = None
    for i, fit in enumerate(fitness_values):
        current += fit
        if current > pick1 and parent1 is None:
            parent1 = population[i]
            break

    current = 0
    parent2 = None
    for i, fit in enumerate(fitness_values):
        current += fit
        if current > pick2:
            parent2 = population[i]
            break

    if parent1 is None:
        parent1 = random.choice(population)
    if parent2 is None:
        parent2 = random.choice(population)

    return parent1, parent2

# 교차: 단일 교차점 방식으로 두 부모로부터 자식 생성
def crossover(parent1, parent2):
    """
    parent1, parent2: 부모 해
    반환값: 자식 해
    """
    point = random.randint(1, len(parent1) - 2)  # 교차점 선택
    child = parent1[:point] + parent2[point:]
    # 교차 후 중복된 값이 있으면 조정
    child = adjust_child(child)
    return child

# 중복된 값을 조정하여 유효한 해로 만듦
def adjust_child(child):
    """
    child: 교차 후 생성된 자식 해
    반환값: 중복이 제거된 유효한 해
    """
    used = set(child)
    missing = set(range(8)) - used  # 누락된 숫자
    for i in range(len(child)):
        if child[i] in child[:i]:  # 중복된 경우
            child[i] = missing.pop()
    return child

# 돌연변이: 낮은 확률로 위치 변경
def mutate(solution, mutation_rate=0.1):
    """
    solution: 돌연변이를 적용할 해
    mutation_rate: 돌연변이 확률
    반환값: 돌연변이가 적용된 해
    """
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(8), 2)  # 두 위치 선택
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]  # 위치 교환
    return solution

# 유전자 알고리즘 메인 함수
def genetic_algorithm(pop_size=100, max_generations=1000):
    """
    pop_size: 개체군 크기
    max_generations: 최대 세대 수
    반환값: 최적 해와 적합도
    """
    # 초기 개체군 생성
    population = create_population(pop_size)
    
    for generation in range(max_generations):
        # 적합도 계산
        fitness_values = [fitness(ind) for ind in population]
        
        # 적합도가 28이면 최적 해 발견
        max_fitness = max(fitness_values)
        if max_fitness == 28:
            best_idx = fitness_values.index(max_fitness)
            return population[best_idx], max_fitness
        
        # 새로운 개체군 생성
        new_population = []
        for _ in range(pop_size):
            # 부모 선택
            parent1, parent2 = select_parents(population, fitness_values)
            # 교차
            child = crossover(parent1, parent2)
            # 돌연변이
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # 세대 정보 출력
        if generation % 100 == 0:
            print(f"세대 {generation}: 최대 적합도 = {max_fitness}")
    
    # 최대 세대에 도달하면 가장 적합한 해 반환
    fitness_values = [fitness(ind) for ind in population]
    best_idx = fitness_values.index(max(fitness_values))
    return population[best_idx], fitness_values[best_idx]

# 결과 출력 함수
def print_board(solution):
    """
    solution: 퀸의 위치를 나타내는 리스트
    체스판에 퀸 위치를 출력
    """
    board = [['.' for _ in range(8)] for _ in range(8)]
    for col, row in enumerate(solution):
        board[row][col] = 'Q'
    for row in board:
        print(' '.join(row))

# 메인 실행
if __name__ == "__main__":
    # 유전자 알고리즘 실행
    best_solution, best_fitness = genetic_algorithm()
    
    # 결과 출력
    print("\n최적 해:")
    print(best_solution)
    print("적합도:", best_fitness)
    print("\n체스판:")
    print_board(best_solution)
