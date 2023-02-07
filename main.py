import math
import random
import sys
import numpy
import pygame

pygame.init()


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

    def __init__(self, x: float, y: float, dx: float, dy: float, gun_damage: int,
                 fly_distance: float = 200, radius: int = 1, speed: int = 5, color: Color = Color.BLACK):
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

        Patron.patrons.append(self)

    def get_data_for_draw(self):
        return Line(self.x, self.y, self.x + (self.dx * self.speed), self.y + (self.dy * self.speed))
        # return Circle(self.color, self.x, self.y, self.radius)

    def calculate_replace_position(self):
        self.distance += self.speed
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class Gun:
    def __init__(self, patron_count: int = 15, fire_speed: float = 60, patron_speed: int = 10,
                 spread: float = 4, fire_distance: float = 200, damage: int = 5):
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
    def __init__(self, gun: Gun,
                 patrons_count: int = 100, heals: int = 10, speed: float = 0.5, rotation_speed: float = 0.5,
                 x: float = 0, y: float = 0, radius: int = 5, color: Color = Color.random_color(),
                 watch_angle: float = 90, watch_distance: int = 200, actual_angle: float = 0):
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

        self.neurons_1 = [[random.uniform(-1, 1) for _ in range(6)] for _ in range(6)]
        self.neurons_2 = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(6)]

    def __tick__(self):
        self.gun.__tick__()

    def get_data_for_draw(self):
        actual_angle = self.actual_angle
        if actual_angle < 0:
            actual_angle += 360
        if self.actual_heals <= 0:
            return [Circle((
                self.color[0] * (self.gun.actual_patron_count / self.gun.max_patron_count),
                self.color[1] * (self.gun.actual_patron_count / self.gun.max_patron_count),
                self.color[2] * (self.gun.actual_patron_count / self.gun.max_patron_count)),
                self.x, self.y, self.radius)]
        else:
            return [Circle((
                self.color[0] * (self.gun.actual_patron_count / self.gun.max_patron_count),
                self.color[1] * (self.gun.actual_patron_count / self.gun.max_patron_count),
                self.color[2] * (self.gun.actual_patron_count / self.gun.max_patron_count)),
                self.x, self.y, self.radius),
                Sector((actual_angle - 45), (actual_angle + 45),
                       self.x - self.watch_distance, self.y - self.watch_distance,
                       self.watch_distance * 2, self.watch_distance * 2)]

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
            Patron(x=self.x + math.cos(math.radians(self.actual_angle)) * self.radius,
                   y=self.y + math.sin(math.radians(self.actual_angle)) * self.radius,
                   dx=math.cos(math.radians(self.actual_angle)),
                   dy=math.sin(math.radians(self.actual_angle)),
                   gun_damage=self.gun.damage,
                   speed=self.gun.patron_speed)

    def reload(self):
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

            is_watch_enemy = -1

            distance = -1
            angle = 0

            for enemy in all_possibles_enemy:

                distance = math.hypot(warrior.x - enemy.x, warrior.y - enemy.y)
                angle = math.atan2(warrior.y - enemy.y, warrior.x - enemy.x)

                if warrior.actual_angle - (warrior.watch_angle / 2) <= angle <= warrior.actual_angle + (
                        warrior.watch_angle / 2) and distance <= warrior.watch_distance:
                    is_watch_enemy = 1

            start_data = [((warrior.actual_heals / warrior.max_heals) - 0.5) * 2,
                          ((warrior.gun.actual_patron_count / warrior.gun.max_patron_count) - 0.5) * 2,
                          ((warrior.actual_patrons_count / warrior.max_patrons_count) - 0.5) * 2,
                          ((warrior.gun.fire_distance / max(distance, 0.00001)) - 0.5) * 2,
                          ((warrior.actual_angle - angle) / 360 - 0.5) * 2,
                          is_watch_enemy]

            first_result = numpy.dot(start_data, warrior.neurons_1)
            first_result = tangoid(first_result)
            second_result = numpy.dot(first_result, warrior.neurons_2)
            second_result = tangoid(second_result)

            warrior.rotate(second_result[0])
            warrior.went(second_result[1], 0)
            if second_result[2] < 0:
                warrior.reload()
            elif second_result[2] > 0:
                warrior.fire()


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
        point = numpy.linalg.solve(M1, v1)

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


team_list = [Team(Color.RED, [Warrior(gun=Gun(patron_speed=5), x=450, y=100, radius=10)]),
             Team(Color.BLUE, [Warrior(gun=Gun(patron_speed=5), x=450, y=800, radius=10)])]

user_warrior = Warrior(gun=Gun(patron_speed=20), x=450, y=225, radius=15, color=Color.GREEN)

while True:
    win.fill(Color.WHITE)
    pygame.display.set_caption(f"{str(round(CLOCK.get_fps(), 3))}  {str(round(CLOCK.get_time(), 3))}")

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    key = pygame.key.get_pressed()

    user_warrior.actual_angle = math.degrees(math.atan2(mouse[1] - user_warrior.y, mouse[0] - user_warrior.x))

    if key[pygame.K_w]:
        user_warrior.went(1, 0)

    if key[pygame.K_s]:
        user_warrior.went(-1, 0)

    if key[pygame.K_a]:
        user_warrior.went(0, -1)

    if key[pygame.K_d]:
        user_warrior.went(0, 1)

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

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                user_warrior.fire()

            if event.button == 3:
                user_warrior.reload()

    user_warrior.__tick__()

    for i in user_warrior.get_data_for_draw():
        i.draw()

    for team in team_list:
        team.__tick__()
        new_team_list = []
        for team_enemy in team_list:
            if team_enemy is not team:
                new_team_list += team_enemy.warriors
        team.calculate_neural_network(new_team_list)

        for drawable in team.get_data_for_draw():
            if isinstance(drawable, list):
                for deep_drawable in drawable:
                    deep_drawable.draw()
            else:
                drawable.draw()

    i = 0

    while len(Patron.patrons) - 1 > i:
        patron = Patron.patrons[i]

        for team in team_list:
            for warrior in team.warriors:

                if Collision.collision_segment_and_circle(warrior.x, warrior.y, warrior.radius,
                                                          patron.x, patron.y,
                                                          patron.x + patron.dx * patron.speed,
                                                          patron.y + patron.dy * patron.speed):
                    try:
                        Patron.patrons.remove(patron)
                    except:
                        pass
                    warrior.actual_heals -= patron.gun_damage
                    continue

        patron.calculate_replace_position()
        patron.get_data_for_draw().draw()
        if patron.distance > patron.fly_distance:
            try:
                Patron.patrons.remove(patron)
            except:
                pass
        else:
            i += 1

    pygame.draw.circle(win, Color.BLACK, mouse, 150, 5)

    pygame.display.flip()
    CLOCK.tick(fps)
