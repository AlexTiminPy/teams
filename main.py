import math
import random
import sys
import numpy
import pygame

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

from sklearn.neural_network import MLPClassifier
import pandas as pd

pygame.init()
my_font = pygame.font.SysFont('Comic Sans MS', 15)

"""снятие ограничений на вывод данных в консоль"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

clf_rotate = MLPClassifier(solver='lbfgs',
                           alpha=1e-5,
                           hidden_layer_sizes=(10, 8),
                           random_state=1,
                           # activation="tanh",
                           max_iter=1000000)

clf_move = MLPClassifier(solver='lbfgs',
                         alpha=1e-5,
                         hidden_layer_sizes=(10, 8),
                         random_state=1,
                         # activation="tanh",
                         max_iter=1000000)

clf_reload_pass_fire = MLPClassifier(solver='lbfgs',
                                     alpha=1e-5,
                                     hidden_layer_sizes=(10, 8),
                                     random_state=1,
                                     # activation="tanh",
                                     max_iter=1000000)

train_dataset = pd.read_csv(r"dataset.csv", sep=";", index_col=False)

y_rotate = train_dataset.drop(
    ["% hp"], axis=1).drop(
    ["% patron"], axis=1).drop(
    ["% all patron"], axis=1).drop(
    ["% dist"], axis=1).drop(
    ["look_at_enemy"], axis=1).drop(
    ["look_at_friend"], axis=1).drop(
    ["enemy_on_left"], axis=1).drop(
    ["move"], axis=1).drop(
    ["reload/pass/fire"], axis=1).drop(
    ["enemy_on_right"], axis=1)

y_move = train_dataset.drop(
    ["% hp"], axis=1).drop(
    ["% patron"], axis=1).drop(
    ["% all patron"], axis=1).drop(
    ["% dist"], axis=1).drop(
    ["look_at_enemy"], axis=1).drop(
    ["look_at_friend"], axis=1).drop(
    ["enemy_on_left"], axis=1).drop(
    ["rotate"], axis=1).drop(
    ["reload/pass/fire"], axis=1).drop(
    ["enemy_on_right"], axis=1)

y_reload_pass_fire = train_dataset.drop(
    ["% hp"], axis=1).drop(
    ["% patron"], axis=1).drop(
    ["% all patron"], axis=1).drop(
    ["% dist"], axis=1).drop(
    ["look_at_enemy"], axis=1).drop(
    ["look_at_friend"], axis=1).drop(
    ["enemy_on_left"], axis=1).drop(
    ["move"], axis=1).drop(
    ["rotate"], axis=1).drop(
    ["enemy_on_right"], axis=1)

X = train_dataset.drop(["rotate"], axis=1).drop(["move"], axis=1).drop(["reload/pass/fire"], axis=1)

# X_train_rotate, X_test_rotate, y_train_rotate, y_test_rotate = \
#     train_test_split(X, y_rotate, test_size=0, random_state=42)
# X_train_move, X_test_move, y_train_move, y_test_move = \
#     train_test_split(X, y_move, test_size=0, random_state=42)
# X_train_reload_pass_fire, X_test_reload_pass_fire, y_train_reload_pass_fire, y_test_reload_pass_fire = \
#     train_test_split(X, y_reload_pass_fire, test_size=0, random_state=42)

clf_rotate.fit(X, y_rotate.values.ravel())
clf_move.fit(X, y_move.values.ravel())
clf_reload_pass_fire.fit(X, y_reload_pass_fire.values.ravel())




class Color:
    BLACK = (40, 40, 40)
    WHITE = (215, 215, 215)
    RED = (200, 50, 50)
    GREEN = (50, 200, 50)
    BLUE = (30, 67, 200)
    YELLOW = (200, 200, 50)
    GRAY = (125, 125, 125)
    List_color = ["RED", "BLUE", 'GREEN', 'GRAY']

    @staticmethod
    def random_color():
        return random.choice(Color.List_color)


class Circle:
    def __init__(self, color, x, y, radius):
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self):
        pygame.draw.circle(win, self.color, [self.x, self.y], self.radius)


class Sector:
    def __init__(self, first_grad, second_grad, x, y, dx, dy):
        self.first_grad = first_grad
        self.second_grad = second_grad
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def draw(self):
        pygame.draw.arc(win, Color.RED, [self.x, self.y, self.dx, self.dy],
                        math.radians(-self.second_grad % 360),
                        math.radians(-self.first_grad % 360), 2)

        pygame.draw.line(win, Color.BLACK,
                         [self.x + self.dx / 2, self.y + self.dy / 2],
                         [(math.cos(math.radians(self.first_grad)) * (self.dx / 2)) + self.x + self.dx / 2,
                          (math.sin(math.radians(self.first_grad)) * (self.dy / 2)) + self.y + self.dy / 2], 2)

        pygame.draw.line(win, Color.BLACK,
                         [self.x + self.dx / 2, self.y + self.dy / 2],
                         [(math.cos(math.radians(self.second_grad)) * (self.dx / 2)) + self.x + self.dx / 2,
                          (math.sin(math.radians(self.second_grad)) * (self.dy / 2)) + self.y + self.dy / 2], 2)


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self):
        pygame.draw.line(win, Color.BLACK, [self.x1, self.y1], [self.x2, self.y2], 1)


WIDTH, HEIGHT = 1800, 900
win = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
fps = 120
is_true = True


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def tangoid(x):
    return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))


class Patron:
    patrons = []

    def __init__(self, father, x: float, y: float, dx: float, dy: float, gun_damage: int,
                 fly_distance: float = 500, radius: int = 1, speed: int = 20, color: Color = Color.BLACK
                 ):
        self.x = x
        self.y = y
        self.dx = dx  # + round(random.uniform(-0.1, 0.1), 3)
        self.dy = dy  # + round(random.uniform(-0.1, 0.1), 3)

        self.gun_damage = gun_damage
        self.fly_distance = fly_distance
        self.distance = 0
        self.speed = speed

        self.radius = radius
        self.color = color

        self.father = father

        Patron.patrons.append(self)

    def get_data_for_draw(self):
        # return Line(self.x, self.y, self.x + (self.dx * self.speed), self.y + (self.dy * self.speed))
        return Circle(self.color, self.x, self.y, self.radius)

    def calculate_replace_position(self):
        self.distance += self.speed
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class Gun:
    def __init__(self, patron_count: int = 15, fire_speed: float = 360, patron_speed: int = 10,
                 spread: float = 4, fire_distance: float = 500, damage: int = 5):
        self.max_patron_count = patron_count
        self.actual_patron_count = patron_count
        self.fire_speed = fire_speed
        self.patron_speed = patron_speed
        self.spread = spread
        self.fire_distance = fire_distance
        self.damage = damage

        self.cooldown = fire_speed
        self.actual_cooldown = fire_speed
        self.is_reloaded = True

    def reload(self):
        if self.actual_cooldown == self.cooldown:
            self.actual_patron_count = 0
            self.actual_cooldown = 0
            self.is_reloaded = False

    def __tick__(self):
        if self.actual_cooldown < self.cooldown:
            self.actual_cooldown += 1
        elif self.actual_cooldown >= self.cooldown and not self.is_reloaded:
            self.actual_patron_count = self.max_patron_count
            self.is_reloaded = True


class Warrior:
    def __init__(self, gun: Gun, n1=None, n2=None, n3=None, team=None,
                 patrons_count: int = 500, heals: int = 1000, speed: float = 0.5, rotation_speed: float = 0.5,
                 x: float = 0, y: float = 0, radius: int = 5, color: Color = Color.random_color(),
                 watch_angle: float = 90, watch_distance: int = 500, actual_angle: float = 0, last_score=0):
        self.gun = gun

        self.max_patrons_count = patrons_count
        self.actual_patrons_count = patrons_count
        self.max_heals = heals
        self.actual_heals = heals
        self.speed = speed
        self.rotation_speed = rotation_speed

        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.radius = radius
        self.color = color

        self.watch_angle = watch_angle
        self.watch_distance = watch_distance
        self.actual_angle = actual_angle

        self.score = 0
        self.team = team

        # deform = abs((last_score - 50000) / 250000)
        #
        # if n1 is None:
        #     self.neurons_1 = [[random.uniform(-0.0001, 0.0001) for _ in range(10)] for _ in range(8)]
        # else:
        #     self.neurons_1 = n1
        #     for i in range(len(self.neurons_1)):
        #         for t in range(len(self.neurons_1[i])):
        #             self.neurons_1[i][t] += random.uniform(-deform, deform)
        #
        # if n2 is None:
        #     self.neurons_2 = [[random.uniform(-0.0001, 0.0001) for _ in range(8)] for _ in range(10)]
        #
        # else:
        #     self.neurons_2 = n2
        #     for i in range(len(self.neurons_2)):
        #         for t in range(len(self.neurons_2[i])):
        #             self.neurons_2[i][t] += random.uniform(-deform, deform)
        #
        # if n3 is None:
        #     self.neurons_3 = [[random.uniform(-0.0001, 0.0001) for _ in range(3)] for _ in range(8)]
        #
        # else:
        #     self.neurons_3 = n3
        #     for i in range(len(self.neurons_2)):
        #         for t in range(len(self.neurons_2[i])):
        #             self.neurons_2[i][t] += random.uniform(-deform, deform)

    def __repr__(self):
        return f"{self.score}"

    def __tick__(self):
        self.gun.__tick__()

    def get_data_for_draw(self):
        actual_angle = self.actual_angle
        if actual_angle < 0:
            actual_angle += 360
        # if self.actual_heals <= 0:
        #     return [Circle((
        #         self.color[0] * (self.gun.actual_patron_count / self.gun.max_patron_count),
        #         self.color[1] * (self.gun.actual_patron_count / self.gun.max_patron_count),
        #         self.color[2] * (self.gun.actual_patron_count / self.gun.max_patron_count)),
        #         self.x, self.y, self.radius)]
        # else:
        # return [Circle(self.color, self.x, self.y, self.radius),
        #         Sector((actual_angle - 45), (actual_angle + 45),
        #                self.x - self.watch_distance, self.y - self.watch_distance,
        #                self.watch_distance * 2, self.watch_distance * 2)]

        text_surface = my_font.render(f"{self.gun.actual_patron_count}/{self.gun.max_patron_count}   "
                                      f"{self.actual_patrons_count}/{self.max_patrons_count}", False, (0, 0, 0))
        win.blit(text_surface, (self.x + 20, self.y))

        return [Circle(self.color, self.x, self.y, self.radius),
                Line(self.x, self.y,
                     self.x + math.cos(math.radians(self.actual_angle)) * self.watch_distance,
                     self.y + math.sin(math.radians(self.actual_angle)) * self.watch_distance)]

    def rotate(self, percent):
        self.actual_angle += self.rotation_speed * percent

    def went(self, percent_forward, percent_sideward):
        dx = math.cos(math.radians(self.actual_angle)) * (self.speed * percent_forward)
        dy = math.sin(math.radians(self.actual_angle)) * (self.speed * percent_forward)
        if 0 < self.x + dx < WIDTH:
            self.x += dx
        if 0 < self.y + dy < HEIGHT:
            self.y += dy

        dx = math.cos(math.radians(self.actual_angle + 90)) * (self.speed * percent_sideward)
        dy = math.sin(math.radians(self.actual_angle + 90)) * (self.speed * percent_sideward)
        if 0 < self.x + dx < WIDTH:
            self.x += dx
        if 0 < self.y + dy < HEIGHT:
            self.y += dy

    def fire(self):

        if self.gun.actual_patron_count > 0:
            self.gun.actual_patron_count -= 1
            Patron(father=self,
                   x=self.x + math.cos(math.radians(self.actual_angle)) * self.radius,
                   y=self.y + math.sin(math.radians(self.actual_angle)) * self.radius,
                   dx=math.cos(math.radians(self.actual_angle)),
                   dy=math.sin(math.radians(self.actual_angle)),
                   gun_damage=self.gun.damage)

    def reload(self):
        if self.actual_patrons_count > 0:
            self.actual_patrons_count -= min(self.gun.max_patron_count, self.actual_patrons_count)
            self.gun.reload()


class Team:
    def __init__(self, color, warriors: []):
        self.color = color
        self.warriors = warriors

        self.count = 0

        for warrior in self.warriors:
            warrior.color = self.color

    def __tick__(self):
        for warrior in self.warriors:
            warrior.__tick__()

    def get_data_for_draw(self):
        return [i.get_data_for_draw() for i in self.warriors]

    def calculate_neural_network(self, all_possibles_enemy):

        for warrior in self.warriors:

            look_at_enemy = 0
            look_at_friend = 0

            enemy_on_left = 0
            enemy_on_right = 0

            min_distance = 100000

            for enemy in all_possibles_enemy:

                if enemy is warrior:
                    continue

                distance = math.hypot(enemy.x - warrior.x, enemy.y - warrior.y)
                angle = math.degrees(math.atan2(enemy.y - warrior.y, enemy.x - warrior.x))
                if angle < 0:
                    angle += 360

                act_angle = warrior.actual_angle
                if act_angle < 0:
                    act_angle += 360

                if act_angle > angle > act_angle - 45 and enemy.team is not warrior.team:
                    enemy_on_left = 1

                if act_angle + 45 > angle > act_angle and enemy.team is not warrior.team:
                    enemy_on_right = 1

                if Collision.collision_segment_and_circle(
                        enemy.x, enemy.y, enemy.radius,
                        warrior.x, warrior.y,
                        warrior.x + math.cos(math.radians(warrior.actual_angle)) * warrior.watch_distance,
                        warrior.y + math.sin(math.radians(warrior.actual_angle)) * warrior.watch_distance):

                    pygame.draw.circle(win, Color.GRAY, [enemy.x, enemy.y], enemy.radius * 2, 3)

                    if abs(distance) < min_distance and abs(distance) < warrior.watch_distance:
                        min_distance = abs(distance)

                        if enemy.team is warrior.team:
                            look_at_friend = 1
                            look_at_enemy = 0

                        else:
                            look_at_enemy = 1
                            look_at_friend = 0

            if not look_at_friend and not look_at_enemy:
                min_distance = 0

            start_data = [[
                (warrior.actual_heals / warrior.max_heals),
                (warrior.gun.actual_patron_count / warrior.gun.max_patron_count),
                (warrior.actual_patrons_count / warrior.max_patrons_count),
                (min_distance / warrior.watch_distance),
                look_at_enemy,
                look_at_friend,
                enemy_on_left,
                enemy_on_right
            ]]

            # first_result = numpy.dot(start_data, warrior.neurons_1)
            # first_result = tangoid(first_result)
            # second_result = numpy.dot(first_result, warrior.neurons_2)
            # second_result = tangoid(second_result)

            rotate = clf_rotate.predict(start_data)

            if rotate < 0:
                warrior.rotate(-1)
            elif rotate > 0:
                warrior.rotate(1)

            move = clf_move.predict(start_data)
            if move < 0:
                warrior.went(-1, 0)
            elif move > 0:
                warrior.went(1, 0)

            reload_pass_fire = clf_reload_pass_fire.predict(start_data)

            if reload_pass_fire < 0:
                warrior.reload()
            elif reload_pass_fire > 0:
                warrior.fire()

            # print(start_data)
            # print(rotate, move, reload_pass_fire)
            # print("---------------")


class Collision:
    @staticmethod
    def collision_segment_and_segment(ax1, ay1, ax2, ay2,
                                      bx1, by1, bx2, by2):
        v1 = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
        v2 = (bx2 - bx1) * (ay2 - by1) - (by2 - by1) * (ax2 - bx1)
        v3 = (ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)
        v4 = (ax2 - ax1) * (by2 - ay1) - (ay2 - ay1) * (bx2 - ax1)
        if (v1 * v2 < 0) and (v3 * v4 < 0):
            return True  # пересекаются
        else:
            return False  # не пересекаются

    @staticmethod
    def collision_segment_and_circle(x, y, radius,
                                     x1, y1,
                                     x2, y2):
        zn1 = x2 - x1
        zn2 = y2 - y1
        chs1 = zn2 * (-x1)
        chs2 = zn1 * (-y1)
        A = zn2
        B = -zn1
        C = chs1 - chs2

        A1 = B
        B1 = -A
        C1 = B * (-x) - A * (-y)

        M1 = numpy.array([[A, B],
                          [A1, B1]])  # Матрица (левая часть системы)
        v1 = numpy.array([-C, -C1])  # Вектор (правая часть системы)
        try:
            point = numpy.linalg.solve(M1, v1)
        except numpy.linalg.LinAlgError:
            return False

        distance = abs(math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2))

        if max(x1, x2) >= point[0] >= min(x1, x2) and \
                max(y1, y2) >= point[1] >= min(y1, y2):

            if distance < radius:
                return True
            elif distance == radius:
                return True
            elif distance > radius:
                return False  # возвращаю колво точек, если не надо поправь на да/нет

        else:
            return False


team_list = [Team(Color.RED, warriors=[]),
             Team(Color.BLUE, warriors=[])]

team_list[0].warriors = \
    [Warrior(team=team_list[0], gun=Gun(patron_speed=5), x=100 * i + 100, y=400, radius=10, color=team_list[0].color)
     for i in range(5)]

team_list[1].warriors = \
    [Warrior(team=team_list[1], gun=Gun(patron_speed=5), x=100 * i + 100, y=500, radius=10, color=team_list[1].color)
     for i in range(5)]

# user_warrior = Warrior(gun=Gun(patron_speed=20), x=450, y=225, radius=15, color=Color.GREEN)

GLOBAL_TICK = 0
GLOBAL_STEP = 600

while True:
    # GLOBAL_TICK += 1
    if GLOBAL_TICK >= GLOBAL_STEP:
        Patron.patrons = []
        GLOBAL_TICK = 0
        for team in team_list:
            rating_list = sorted(team.warriors, key=lambda x: x.score, reverse=True)
            warrior_best = rating_list[0]

            yy = random.randint(200, 700)

            team.warriors = [Warrior(
                team=team,
                color=team.color,
                gun=Gun(patron_speed=5),
                x=100 * i + 50, y=yy, radius=10) for i in range(1)]

            print(rating_list)

    win.fill(Color.WHITE)

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    key = pygame.key.get_pressed()

    # user_warrior.actual_angle = math.degrees(math.atan2(mouse[1] - user_warrior.y, mouse[0] - user_warrior.x))

    # if key[pygame.K_w]:
    #     user_warrior.went(1, 0)
    #
    # if key[pygame.K_s]:
    #     user_warrior.went(-1, 0)
    #
    # if key[pygame.K_a]:
    #     user_warrior.went(0, -1)
    #
    # if key[pygame.K_d]:
    #     user_warrior.went(0, 1)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_1:
                t = Warrior(gun=Gun(patron_speed=5), x=mouse[0], y=mouse[1], radius=10)
                t.color = team_list[0].color
                team_list[0].warriors.append(t)

            if event.key == pygame.K_2:
                t = Warrior(gun=Gun(patron_speed=5), x=mouse[0], y=mouse[1], radius=10)
                t.color = team_list[1].color
                team_list[1].warriors.append(t)

            if event.key == pygame.K_SPACE:
                GLOBAL_TICK = GLOBAL_STEP

        if event.type == pygame.MOUSEBUTTONDOWN:
            # user_warrior.fire()
            # user_warrior.reload()
            if event.button == 1:
                for i in team_list:
                    for t in i.warriors:
                        if abs(math.hypot(mouse[0] - t.x, mouse[1] - t.y)) < 50:
                            t.score += 5000

            if event.button == 3:
                for i in team_list:
                    for t in i.warriors:
                        if abs(math.hypot(mouse[0] - t.x, mouse[1] - t.y)) < 50:
                            t.score -= 5000

            if event.button == 4:
                GLOBAL_STEP += 100

            if event.button == 5:
                GLOBAL_STEP -= 100

    pygame.display.set_caption(f"{GLOBAL_STEP}")

    # user_warrior.__tick__()
    #
    # for i in user_warrior.get_data_for_draw():
    #     i.draw()

    for team in team_list:
        team.__tick__()
        new_team_list = []
        for team_enemy in team_list:
            new_team_list += team_enemy.warriors
        team.calculate_neural_network(new_team_list)

        for drawable in team.get_data_for_draw():
            if isinstance(drawable, list):
                for deep_drawable in drawable:
                    deep_drawable.draw()
            else:
                drawable.draw()

    all_possibles_enemy = []
    for team_enemy in team_list:
        all_possibles_enemy += team_enemy.warriors

    # if user_warrior.actual_angle < 0:
    #     pygame.display.set_caption(f"{user_warrior.actual_angle + 360}")
    # else:
    #     pygame.display.set_caption(f"{user_warrior.actual_angle}")
    #
    # for enemy in all_possibles_enemy:
    #
    #     angle = math.degrees(math.atan2(enemy.y - user_warrior.y, enemy.x - user_warrior.x))
    #     if angle < 0:
    #         angle += 360
    #     distance = math.hypot(user_warrior.x - enemy.x, user_warrior.y - enemy.y)
    #
    #     act_angle = user_warrior.actual_angle
    #     if act_angle < 0:
    #         act_angle += 360
    #
    #     if act_angle > angle > act_angle - 45 and abs(
    #             distance) < user_warrior.watch_distance:
    #         pygame.draw.circle(win, Color.GRAY, [enemy.x, enemy.y], 10, 2)
    #
    #     if act_angle + 45 > angle > act_angle and abs(
    #             distance) < user_warrior.watch_distance:
    #         pygame.draw.circle(win, Color.BLACK, [enemy.x, enemy.y], 10, 2)

    i = 0

    while len(Patron.patrons) - 1 > i:
        patron = Patron.patrons[i]

        for team in team_list:
            for warrior in team.warriors:

                if Collision.collision_segment_and_circle(warrior.x, warrior.y, warrior.radius,
                                                          patron.x, patron.y,
                                                          patron.x + patron.dx * patron.speed,
                                                          patron.y + patron.dy * patron.speed):
                    if patron.father.team is team:
                        patron.father.score -= 400
                    else:
                        patron.father.score += 100
                    try:
                        Patron.patrons.remove(patron)
                    except:
                        pass
                    warrior.actual_heals -= patron.gun_damage
                    warrior.score -= 50
                    continue

        patron.calculate_replace_position()
        patron.get_data_for_draw().draw()
        if patron.distance > patron.fly_distance:
            try:
                patron.father.score -= 1
                Patron.patrons.remove(patron)
            except:
                pass
        else:
            i += 1

    pygame.draw.circle(win, Color.BLACK, mouse, 50, 5)

    pygame.display.flip()
    CLOCK.tick(fps)
