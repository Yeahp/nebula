from dateutil import parser
import time


if __name__ == "__main__":
    """
    To construct a specific time, we need assign 9 parameters:
    year, month, day, hour, minute, second, weekday, yearday, isdst
    """

    start_1, start_2 = time.clock(), time.time_prac()

    t_tuple = (2018, 11, 11, 12, 10, 9, 7, 315, 0)
    t_time_1 = time.mktime(t_tuple)  # return a timestamp, float type
    #print("t_time_1: ", t_time_1)

    t_time_2 = time.localtime(t_time_1)  # return a structure_time
    #print("t_time_2:", type(t_time_2))

    t_time_3 = time.asctime(t_time_2)  # return a string
    #print("t_time_3: ", t_time_3)

    t_time_4 = parser.parse(t_time_3)  #
    #print("t_time_4: ", t_time_4)

    #print(time.strftime("%b %d %Y %H:%M:%S", time.localtime(time.time())))
    #print(time.strftime("%b %d %Y %H:%M:%S", time.gmtime(time.time())))
    print(time.strftime("%Y-%m-%d"))

    end_1, end_2 = time.clock(), time.time()
    #print("time used 1: ", end_1 - start_1)
    #print("time used 2: ", end_2 - start_2)

    """
    %y 两位数的年份表示（00-99）
    %Y 四位数的年份表示（000-9999）
    %m 月份（01-12）
    %d 月内中的一天（0-31）
    %H 24小时制小时数（0-23）
    %I 12小时制小时数（01-12）
    %M 分钟数（00=59）
    %S 秒（00-59）
    %a 本地简化星期名称
    %A 本地完整星期名称
    %b 本地简化的月份名称
    %B 本地完整的月份名称
    %c 本地相应的日期表示和时间表示
    %j 年内的一天（001-366）
    %p 本地A.M.或P.M.的等价符
    %U 一年中的星期数（00-53）星期天为星期的开始
    %w 星期（0-6），星期天为星期的开始
    %W 一年中的星期数（00-53）星期一为星期的开始
    %x 本地相应的日期表示
    %X 本地相应的时间表示
    %Z 当前时区的名称
    %% %号本身
    """
