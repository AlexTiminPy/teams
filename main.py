import copy
import math
import random
import sys
import numpy
import pygame
import pickle

import warnings

from collections import namedtuple

warnings.filterwarnings(action='ignore', category=UserWarning)

pygame.init()


def get_models():
    with open('modelsSave/rotate_model.pkl', 'rb') as f:
        clf_rotate = pickle.load(f)

    with open('modelsSave/move_model.pkl', 'rb') as f:
        clf_move = pickle.load(f)

    with open('modelsSave/reload_pass_fire_model.pkl', 'rb') as f:
        clf_reload_pass_fire = pickle.load(f)

    return clf_rotate, clf_move, clf_reload_pass_fire


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
    def __init__(self, patron_count: int = 15, fire_speed: float = 360, patron_speed: int = 20,
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


class ExternalPartWarrior:
    def __init__(self, x: float = 0, y: float = 0, radius: int = 3, color: Color = Color.random_color()):
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.radius = radius
        self.color = color

    def get_data_for_draw(self):
        return [Circle(self.color, self.x, self.y, self.radius)]


class InternalPartWarrior:
    def __init__(self):
        self.rotate_model = DecisionMakingWarriors.mutate(clf_rotate)
        self.move_model = DecisionMakingWarriors.mutate(clf_move)
        self.reload_pass_fire_model = DecisionMakingWarriors.mutate(clf_reload_pass_fire)


class FightPartWarrior:
    def __init__(self, patrons_count: int = 500, heals: int = 1, speed: float = 0.5, rotation_speed: float = 1,
                 watch_angle: float = 90, watch_distance: int = 500):
        self.max_patrons_count = patrons_count
        self.actual_patrons_count = patrons_count
        self.max_heals = heals
        self.actual_heals = heals
        self.speed = speed
        self.rotation_speed = rotation_speed

        self.watch_angle = watch_angle
        self.watch_distance = watch_distance
        self.actual_angle = random.randint(0, 360)


class Team:
    def __init__(self, name, color):
        self.name = name
        self.color = color


class Warrior:
    warriors = []

    def __init__(self, gun: Gun, team: Team, external: ExternalPartWarrior, internal: InternalPartWarrior, fight: FightPartWarrior):
        self.gun = gun

        self.team = team

        self.external = external
        self.internal = internal
        self.fight = fight

        Warrior.warriors.append(self)

    def __tick__(self):
        if self.gun.actual_cooldown < self.gun.cooldown:
            self.gun.actual_cooldown += 1
        elif self.gun.actual_cooldown >= self.gun.cooldown and not self.gun.is_reloaded:
            self.actual_patron_count = self.gun.max_patron_count
            self.is_reloaded = True

    def get_data_for_draw(self):

        return self.external.get_data_for_draw()

    def rotate(self, percent):
        self.fight.actual_angle += self.fight.rotation_speed * percent

    def went(self, percent_forward):
        dx = math.cos(math.radians(self.fight.actual_angle)) * (self.fight.speed * percent_forward)
        dy = math.sin(math.radians(self.fight.actual_angle)) * (self.fight.speed * percent_forward)
        if 0 < self.external.x + dx < WIDTH:
            self.external.x += dx
        if 0 < self.external.y + dy < HEIGHT:
            self.external.y += dy

    def fire(self):

        if self.gun.actual_patron_count > 0:
            self.gun.actual_patron_count -= 1
            Patron(father=self,
                   x=self.external.x + math.cos(math.radians(self.fight.actual_angle)) * self.external.radius,
                   y=self.external.y + math.sin(math.radians(self.fight.actual_angle)) * self.external.radius,
                   dx=math.cos(math.radians(self.fight.actual_angle)),
                   dy=math.sin(math.radians(self.fight.actual_angle)),
                   gun_damage=self.gun.damage)

    def reload(self):
        if self.fight.actual_patrons_count < 0:
            return

        self.fight.actual_patrons_count -= min(self.gun.max_patron_count, self.fight.actual_patrons_count)

        if self.gun.actual_cooldown == self.gun.cooldown:
            self.gun.actual_patron_count = 0
            self.gun.actual_cooldown = 0
            self.gun.is_reloaded = False


class DecisionMakingWarriors:
    @staticmethod
    def mutate(model):
        new_model = copy.deepcopy(model)
        for cf in new_model.coefs_:
            for i in range(len(cf)):
                for t in range(len(cf[i])):
                    cf[i][t] += random.uniform(-0.01, 0.01)

        return new_model

    @staticmethod
    def calculate_neural_network(warrior, all_possibles_enemy):

        start_data = DecisionMakingWarriors.get_start_data(warrior, all_possibles_enemy)
        rotate, move, reload_pass_fire = DecisionMakingWarriors.get_predicts(warrior, start_data)
        DecisionMakingWarriors.activate(warrior, rotate, move, reload_pass_fire)

    @staticmethod
    def get_start_data(warrior, all_possibles_enemy):
        look_at_friend = 0
        look_at_enemy = 0

        enemy_on_left = 0
        enemy_on_right = 0

        min_distance = 100000

        for enemy in all_possibles_enemy:

            if enemy is warrior:
                continue

            distance = abs(math.hypot(enemy.external.x - warrior.external.x, enemy.external.y - warrior.external.y))

            if distance > warrior.fight.watch_distance:
                continue

            min_distance = min(min_distance, distance)

            enemy_on_left, enemy_on_right = DecisionMakingWarriors.get_left_right_enemy(warrior, enemy, enemy_on_left, enemy_on_right)

            look_at_enemy, look_at_friend = DecisionMakingWarriors.get_is_look_on(warrior, enemy)

        if not look_at_friend and not look_at_enemy:
            min_distance = 0

        return [
            DecisionMakingWarriors.get_easy_data(warrior, min_distance) +
            [
                look_at_enemy,
                look_at_friend,
                int(bool(enemy_on_left)),
                int(bool(enemy_on_right))
            ]]

    @staticmethod
    def get_is_look_on(warrior, enemy):
        look_at_enemy = 0
        look_at_friend = 0

        if not Collision.collision_segment_and_circle(
                enemy.external.x, enemy.external.y, enemy.external.radius,
                warrior.external.x, warrior.external.y,
                warrior.external.x + math.cos(math.radians(warrior.fight.actual_angle)) * warrior.fight.watch_distance,
                warrior.external.y + math.sin(math.radians(warrior.fight.actual_angle)) * warrior.fight.watch_distance):
            return look_at_enemy, look_at_friend

        if enemy.team is warrior.team:
            look_at_friend = 1
            look_at_enemy = 0
            return look_at_enemy, look_at_friend

        look_at_enemy = 1
        look_at_friend = 0

        return look_at_enemy, look_at_friend

    @staticmethod
    def get_easy_data(warrior, min_distance):
        return [warrior.fight.actual_heals / warrior.fight.max_heals,
                warrior.gun.actual_patron_count / warrior.gun.max_patron_count,
                warrior.fight.actual_patrons_count / warrior.fight.max_patrons_count,
                (min_distance / warrior.fight.watch_distance)]

    @staticmethod
    def get_left_right_enemy(warrior, enemy, last_enemy_on_left, last_enemy_on_right):
        enemy_on_left = last_enemy_on_left
        enemy_on_right = last_enemy_on_right
        angle = math.degrees(math.atan2(enemy.external.y - warrior.external.y, enemy.external.x - warrior.external.x))
        angle = max(angle, angle + 360)

        act_angle = max(warrior.fight.actual_angle, warrior.fight.actual_angle + 360)

        if act_angle > angle > act_angle - 45:
            enemy_on_left += 1

        elif act_angle + 45 > angle > act_angle:
            enemy_on_right += 1

        return enemy_on_left, enemy_on_right

    @staticmethod
    def activate(warrior, rotate, move, reload_pass_fire):

        warrior.rotate(rotate[0])

        warrior.went(move[0])

        if reload_pass_fire < 0:
            warrior.reload()
        elif reload_pass_fire > 0:
            warrior.fire()

    @staticmethod
    def get_predicts(warrior, start_data):
        rotate = warrior.internal.rotate_model.predict(start_data)
        move = warrior.internal.move_model.predict(start_data)
        reload_pass_fire = warrior.internal.reload_pass_fire_model.predict(start_data)

        return rotate, move, reload_pass_fire


class Collision:
    @staticmethod
    def collision_segment_and_segment(ax1, ay1, ax2, ay2,
                                      bx1, by1, bx2, by2):
        v1 = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
        v2 = (bx2 - bx1) * (ay2 - by1) - (by2 - by1) * (ax2 - bx1)
        v3 = (ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)
        v4 = (ax2 - ax1) * (by2 - ay1) - (ay2 - ay1) * (bx2 - ax1)
        if (v1 * v2 < 0) and (v3 * v4 < 0):
            return True
        else:
            return False

    @staticmethod
    def collision_segment_and_circle(x, y, radius,
                                     x1, y1,
                                     x2, y2):
        if abs(math.hypot(abs(x - x1), abs(y - y1))) < abs(math.hypot(abs(x2 - x1), abs(y2 - y1))):
            return False

        try:
            point = numpy.linalg.solve(numpy.array([[y2 - y1, x1 - x2],
                                                    [x1 - x2, y1 - y2]]),
                                       numpy.array([(x2 - x1) * -y1 - (y2 - y1) * -x1,
                                                    (y2 - y1) * -y - (x2 - x1) * x]))
        except numpy.linalg.LinAlgError:
            return False

        distance = math.hypot(point[0] - x, point[1] - y)

        if max(x1, x2) >= point[0] >= min(x1, x2) and \
                max(y1, y2) >= point[1] >= min(y1, y2):

            return distance <= radius

        else:
            return False


WIDTH, HEIGHT = 1800, 900
win = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
fps = 120
is_true = True

my_font = pygame.font.SysFont('Comic Sans MS', 15)

clf_rotate, clf_move, clf_reload_pass_fire = get_models()

team1 = Team("Goffy", Color.RED)
team2 = Team("Micky", Color.BLUE)

for i in range(5):
    Warrior(gun=Gun(),
            team=team1,
            external=ExternalPartWarrior(x=100 * i + 100, y=400, color=team1.color),
            internal=InternalPartWarrior(),
            fight=FightPartWarrior())
    Warrior(gun=Gun(),
            team=team2,
            external=ExternalPartWarrior(x=100 * i + 100, y=500, color=team2.color),
            internal=InternalPartWarrior(),
            fight=FightPartWarrior())

GLOBAL_TICK = 0
GLOBAL_STEP = 3600

while True:
    pygame.display.set_caption(f"{CLOCK.get_fps()}")

    win.fill(Color.WHITE)

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    key = pygame.key.get_pressed()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:
                for i in range(3):
                    for t in range(3):
                        Warrior(gun=Gun(),
                                team=team1,
                                external=ExternalPartWarrior(x=mouse[0] + 25 * i, y=mouse[1] + 25 * t, color=team1.color),
                                internal=InternalPartWarrior(),
                                fight=FightPartWarrior())

            if event.button == 3:
                for i in range(3):
                    for t in range(3):
                        Warrior(gun=Gun(),
                                team=team2,
                                external=ExternalPartWarrior(x=mouse[0] + 25 * i, y=mouse[1] + 25 * t, color=team2.color),
                                internal=InternalPartWarrior(),
                                fight=FightPartWarrior())

    for warrior in Warrior.warriors:
        warrior.__tick__()

        DecisionMakingWarriors.calculate_neural_network(warrior, Warrior.warriors)

        for drawable in warrior.get_data_for_draw():
            drawable.draw()

    i = 0

    while len(Patron.patrons) - 1 > i:
        patron = Patron.patrons[i]

        for warrior in Warrior.warriors:

            if not Collision.collision_segment_and_circle(warrior.external.x, warrior.external.y, warrior.external.radius,
                                                          patron.x, patron.y,
                                                          patron.x + patron.dx * patron.speed,
                                                          patron.y + patron.dy * patron.speed):
                continue

            try:
                Patron.patrons.remove(patron)
            except:
                pass
            warrior.fight.actual_heals -= patron.gun_damage

            if warrior.fight.actual_heals <= 0:
                try:
                    Warrior.warriors.remove(warrior)
                except:
                    pass

        patron.calculate_replace_position()
        patron.get_data_for_draw().draw()
        if patron.distance > patron.fly_distance:
            try:
                Patron.patrons.remove(patron)
            except:
                pass
        else:
            i += 1

    pygame.draw.circle(win, Color.BLACK, mouse, 50, 5)

    pygame.display.flip()
    CLOCK.tick(fps)
