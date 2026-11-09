# 3D-A 09e — unified contact и ref_planes

Источник задания: docs/coord/3DA_PROPS_2026-11-09e.md, пункты0b/0c. Факты и SHA256: [JSON](3DA_CONTACT_SPEED_2026-11-09e.json).

Контакт: хотя бы одна вершина declared support_parts в ±YAML contact_tol; существующие медианные зазоры low-band групп сохранены. >contact_tol: support_gap со списком/максимумом в rep.support_contact; >0.05м: partial. Другие geometry failures видны отдельными checks/flags; contact_fail зарезервирован отсутствию контакта. wrong_surface/out_of_frame сохранены; нового enum нет. Ранний zone_empty не имеет тела и остаётся неподдерживаемым для H3, даже если status partial: это не статусный блок, отсутствует геометрия.

0c: bounded numpy RANSAC scoring с прежним RNG/order/ties, content-keyed memo только внутри placement. Cached refs Q01/Q03/Q04/T02: X/KP/planes max abs delta **0** (допуск1e-6), penetration/status совпали при одинаковой новой0b. CPU old→new: Q01 **4.636→3.128с**; Q03 **494.274→510.987с**; Q04 **5.270→3.627с**; T02 **6.047→5.669с**. Статусы ok/ok/partial/wrong_surface. Это не сравнение разных контактных правил и не live generic evidence.

Один after CPU cProfile Q01: place3.127603с; ref_planes2.211124с=**70.6971%**; prepare_body2.230948с. Реальный ref_planes call1, поэтому benefit повторного memo не приписывается measured place. Refinement/model loader/GPU/Blender/H3 в профиле отсутствуют. SLA≤60с не подтверждён для всех: Q03≈511с. Raw profile: C:/work/scene3d_work/farm/diagnostics_09e/Q01_after_0c_cpu.pstats.

Targeted72pass+19subtests; дополнительная long группа24pass/1obsolete fail; obsolete lie head-height test обновлён подv0.16(axis75–105°), нужный метод1pass3.78с. Long full повторно не запускался; итоговый совокупный тестовый счёт — в основном отчёте. В этих CPU фактах нет GPU прогона. Generic Q04 с новым контактом дополнительно реально запускался main: partial, placement52.641с,total65.344с (параллельно runner); props версия placement45.187с,total59.672с. Не выдавать эти разные участки за гарантированное ускорение173→<60 на всех позах.
