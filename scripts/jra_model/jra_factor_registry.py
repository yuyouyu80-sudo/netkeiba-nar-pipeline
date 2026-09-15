# -*- coding: utf-8 -*-
"""ファクター検証データベースの「ファクター」定義を1箇所に集約する辞書。

「グループ→個別オプション」の2階層。新しいファクターを追加する場合はこの辞書に
1エントリ足すだけでよい(JSON化してArtifactへそのまま埋め込む設計、build_factor_dataset.py
とbuild_factor_artifact.pyの両方がこの辞書だけを参照する)。

`widget`(UIレビュー反映、2026-08-31): 累積的な順位カットオフ(「1位のみ」⊂「1〜3位」)は
複数選択のOR条件にすると無意味/紛らわしいため`"radio"`(単一選択、「絞り込みなし」が既定)
にする。真にOR意味のあるフラットなカテゴリ(レース種別、予想印の個別マーク)だけ
`"checkbox"`(複数選択)にする。`kind`は評価方法(`rank_le`/`threshold_ge`/`category_in`)、
`widget`は見た目(UIレビューの指摘反映)。

欠損値の意味論(2026-09-03、シニアエンジニアレビュー指摘により明文化): `rank_le`/
`threshold_ge`系ソース列(pred_rank・career_place_rate等)は欠損を素の`None`で表す。
`category_in`系ソース列(grade_best・layoff_flag等)は`None`ではなく明示的なセンチネル
文字列("unknown"/"no_history"/"insufficient_history"等)で欠損を表す。いずれの表現も
`FACTOR_GROUPS`の`options`に対応するidを登録していないため、UI上は選択不可=常に非該当
(絞り込み対象外)として扱われ、挙動としては統一されている。
"""

FACTOR_GROUPS = {
    "race_type": {
        "label": "レース種別", "source": "race_type", "kind": "category_in", "widget": "checkbox",
        "options": [
            {"id": "rt_normal", "label": "通常戦", "params": {"in": ["normal"]}},
            {"id": "rt_shinba", "label": "新馬戦", "params": {"in": ["shinba"]}},
            {"id": "rt_mishoubi", "label": "未勝利戦", "params": {"in": ["mishoubi"]}},
        ],
    },
    "score_rank": {
        "label": "本番予想スコア順位(BOX5モデル基準)",
        "source": "pred_rank", "kind": "rank_le", "widget": "radio", "tier": "basic",
        "options": [
            {"id": "score_top1", "label": "1位のみ", "params": {"le": 1}},
            {"id": "score_top2", "label": "1〜2位", "params": {"le": 2}},
            {"id": "score_top3", "label": "1〜3位", "params": {"le": 3}},
            {"id": "score_top4", "label": "1〜4位", "params": {"le": 4}},
            {"id": "score_top5", "label": "1〜5位", "params": {"le": 5}},
            {"id": "score_top6", "label": "1〜6位", "params": {"le": 6}},
        ],
    },
    "score_rank_box4": {
        "label": "本番予想スコア順位(BOX4モデル基準)",
        "source": "pred_rank_box4", "kind": "rank_le", "widget": "radio", "tier": "reference",
        "options": [
            {"id": "score_box4_top1", "label": "1位のみ", "params": {"le": 1}},
            {"id": "score_box4_top2", "label": "1〜2位", "params": {"le": 2}},
            {"id": "score_box4_top3", "label": "1〜3位", "params": {"le": 3}},
            {"id": "score_box4_top4", "label": "1〜4位", "params": {"le": 4}},
            {"id": "score_box4_top5", "label": "1〜5位", "params": {"le": 5}},
            {"id": "score_box4_top6", "label": "1〜6位", "params": {"le": 6}},
        ],
        "extra_note": "2026-09-03、参考指標タイアへ移動(UI/UXレビュー指摘反映。BOX5(score_rank、"
                      "基本タイア)と並列に「基本」扱いされていると、本番採用モデルと誤認され"
                      "やすいため)。通常戦(246レース)のみ対象、新馬戦・未勝利戦はBOX4モデル"
                      "自体が存在しないため常にNaN(対象外)。BOX4/BOX3の現行重みは2026-08-11〜12の"
                      "専門家レビューでNested LOBO OOF未検証と判明し、再探索(25シグナル・"
                      "3サイズ)も市場割れで不採用となった経緯があります"
                      "(詳細はproject_jra_box4box3_degradation_root_cause_2026_08_12参照)。"
                      "BOX5より参考程度に見てください。",
    },
    "score_rank_box3": {
        "label": "本番予想スコア順位(BOX3モデル基準)",
        "source": "pred_rank_box3", "kind": "rank_le", "widget": "radio", "tier": "reference",
        "options": [
            {"id": "score_box3_top1", "label": "1位のみ", "params": {"le": 1}},
            {"id": "score_box3_top2", "label": "1〜2位", "params": {"le": 2}},
            {"id": "score_box3_top3", "label": "1〜3位", "params": {"le": 3}},
            {"id": "score_box3_top4", "label": "1〜4位", "params": {"le": 4}},
            {"id": "score_box3_top5", "label": "1〜5位", "params": {"le": 5}},
            {"id": "score_box3_top6", "label": "1〜6位", "params": {"le": 6}},
        ],
        "extra_note": "score_rank_box4と同じ注意点(通常戦のみ対象、現行重みは未検証、"
                      "2026-09-03に参考指標タイアへ移動)。",
    },
    "mark_honshi": {
        "label": "予想印(本紙)",
        "source": "mark_honshi", "kind": "category_in", "widget": "checkbox", "tier": "basic",
        "options": [
            {"id": "mark_honmei", "label": "◎(本命)", "params": {"in": ["◎"]}},
            {"id": "mark_taikou", "label": "○(対抗)", "params": {"in": ["○", "〇"]}},
            {"id": "mark_tanana", "label": "▲(単穴)", "params": {"in": ["▲"]}},
            {"id": "mark_renka", "label": "△(連下)", "params": {"in": ["△"]}},
            {"id": "mark_chuumoku", "label": "☆(注目)", "params": {"in": ["☆"]}},
        ],
        "extra_note": "2026-09-03修正: 旧「×(注意)」は実データに1件も出現しない記号だった"
                      "(mark_list_parser.pyが実地確認したのは◎○▲△☆の5種類のみ)ため"
                      "「☆(注目)」に訂正しました。",
    },
    "mark_cp": {
        "label": "予想印(CP)",
        "source": "mark_cp", "kind": "category_in", "widget": "checkbox", "tier": "basic",
        "options": [
            {"id": "mark_cp_honmei", "label": "◎(本命)", "params": {"in": ["◎"]}},
            {"id": "mark_cp_taikou", "label": "○(対抗)", "params": {"in": ["○", "〇"]}},
            {"id": "mark_cp_tanana", "label": "▲(単穴)", "params": {"in": ["▲"]}},
            {"id": "mark_cp_renka", "label": "△(連下)", "params": {"in": ["△"]}},
        ],
        "extra_note": "2026-09-03追加(ユーザー依頼)。「CP予想」(netkeiba側のコンピュータ予想)"
                      "の印。実データでは☆(注目)の使用が確認できなかったため◎○▲△の4種類のみ。",
    },
    "mark_other": {
        "label": "予想印(その他専門家、集計)",
        "source": "mark_other", "kind": "category_in", "widget": "checkbox", "tier": "basic",
        "options": [
            {"id": "mark_other_honmei", "label": "◎(本命)", "params": {"in": ["◎"]}},
            {"id": "mark_other_taikou", "label": "○(対抗)", "params": {"in": ["○", "〇"]}},
            {"id": "mark_other_tanana", "label": "▲(単穴)", "params": {"in": ["▲"]}},
            {"id": "mark_other_renka", "label": "△(連下)", "params": {"in": ["△"]}},
        ],
        "extra_note": "本紙・CPを除く全専門家の印を配点集計(◎=6/○=4/▲=3/△=2/☆=0.5点)し、"
                      "順位1〜4位グループを◎○▲△として再割当した合成値(生の個別専門家印では"
                      "ない)。全専門家が無印の馬(★)は「参考指標」タイア内の別グループ"
                      "「予想印: 全専門家無印」に独立させました(2026-09-03、ユーザー指摘反映)。",
    },
    "no_mark_all": {
        "label": "予想印: 全専門家無印",
        "source": "mark_other", "kind": "category_in", "widget": "checkbox", "tier": "reference",
        "options": [{"id": "no_mark_all_yes", "label": "本紙・CP・その他全専門家が無印",
                     "params": {"in": ["★"]}}],
        "extra_note": "参考・不採用指標。本紙・CPだけでなくその他の全専門家(mark_raw_*列すべて)"
                      "も含め、印が1つも付いていない馬だけに立つフラグ(mark_list_parser.py "
                      "summarize_marks()、scoreが0点=その他全専門家も無印、かつ本紙・CPも空欄の"
                      "場合のみ★になる)。プラス評価の◎○▲△(「基本」タイア内の「予想印(その他"
                      "専門家、集計)」グループ)とは独立した「無評価」カテゴリで、凡走予測(flop)と"
                      "同様、本番モデルには使われていません。",
    },
    "ninki": {
        "label": "市場人気(単勝人気、モデル非依存)",
        "source": "bias_ninki", "kind": "rank_le", "widget": "radio", "tier": "basic",
        "options": [
            {"id": "ninki_1", "label": "1番人気のみ", "params": {"le": 1}},
            {"id": "ninki_top3", "label": "1〜3番人気", "params": {"le": 3}},
            {"id": "ninki_top5", "label": "1〜5番人気", "params": {"le": 5}},
        ],
    },
    "radar_ability": {
        "label": "能力・調教評価 順位", "source": "radar_rank_ability",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_ability_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_ability_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_ability_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_ability_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_ability_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_ability_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    "radar_form": {
        "label": "近走成績・調子 順位", "source": "radar_rank_form",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_form_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_form_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_form_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_form_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_form_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_form_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    "radar_style": {
        "label": "脚質・展開 順位", "source": "radar_rank_style",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_style_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_style_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_style_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_style_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_style_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_style_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    "radar_aptitude": {
        "label": "コース・距離適性 順位", "source": "radar_rank_aptitude",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_aptitude_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_aptitude_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_aptitude_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_aptitude_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_aptitude_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_aptitude_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    "radar_pedigree": {
        "label": "血統適性 順位", "source": "radar_rank_pedigree",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_pedigree_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_pedigree_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_pedigree_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_pedigree_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_pedigree_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_pedigree_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    "radar_jt": {
        "label": "騎手・厩舎 順位", "source": "radar_rank_jt",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_jt_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_jt_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_jt_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_jt_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_jt_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_jt_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    "radar_mark": {
        "label": "予想印・専門家評価 順位", "source": "radar_rank_mark",
        "kind": "rank_le", "widget": "radio", "tier": "radar",
        "options": [{"id": "radar_mark_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_mark_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_mark_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_mark_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_mark_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_mark_top6", "label": "1〜6位", "params": {"le": 6}}],
    },
    # =====================================================================
    # レーダー細分化フィルタ(2026-09-14追加、レーダー解像度レビュー
    # https://claude.ai/code/artifact/6744447f-b10c-4c77-a344-bf50755a47fd をOpus5
    # サブエージェントに調査させ、★3件+条件付き1件を採用)。既存「詳細7カテゴリ(レーダー)」
    # タイア(radar_ability〜radar_mark、7項目)は無変更のまま、「脚質・展開」「騎手・厩舎」
    # 「血統適性」の3カテゴリをより細かい9本のサブ軸へ分割。jra_radar_categories.pyの
    # DISPLAY_SUBCATEGORY_MAP(CATEGORY_SIGNAL_MAP/ORDER_1/2/NULL・既に不採用のレーダー
    # 面積予想手法とは完全に独立)から算出。
    # =====================================================================
    "radar_style_position": {
        "label": "脚質・位置取り 順位", "source": "radar_rank_style_position",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_style_position_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_style_position_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_style_position_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_style_position_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_style_position_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_style_position_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「脚質・展開」(詳細7カテゴリ)の細分軸その1。style/nige/corner3・4"
                      "コーナーの位置取り・ギャップ(6シグナル)のみで構成、相互に独立な"
                      "3分割の1本目(rho≤0.15)。既存「脚質・展開 順位」(radar_style、"
                      "13シグナルの集約)とは別軸で、そちらは無変更のまま残しています。",
    },
    "radar_style_stamina": {
        "label": "持続力(スタミナ) 順位", "source": "radar_rank_style_stamina",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_style_stamina_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_style_stamina_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_style_stamina_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_style_stamina_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_style_stamina_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_style_stamina_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「脚質・展開」の細分軸その2。holdtime/hold_just/hold_wide(持続タイム系"
                      "3シグナル)のみで構成。",
    },
    "radar_style_corner_move": {
        "label": "コーナーでの押し上げ 順位", "source": "radar_rank_style_corner_move",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_style_corner_move_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_style_corner_move_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_style_corner_move_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_style_corner_move_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_style_corner_move_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_style_corner_move_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「脚質・展開」の細分軸その3。corner4_speedup/corner_transition_rank・"
                      "gap(コーナー間の順位変化・押し上げ系3シグナル)のみで構成。なお"
                      "pace_fit(想定ペース×脚質)はこの3分割から意図的に除外しています"
                      "(カバレッジ58%・約半数がタイで順位フィルタとして機能しないため。"
                      "絞り込みたい場合は既存の「想定ペース」「4コーナー想定位置」フィルタを"
                      "使ってください)。",
    },
    "radar_jt_base": {
        "label": "騎手・厩舎: 素の成績 順位", "source": "radar_rank_jt_base",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_jt_base_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_jt_base_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_jt_base_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_jt_base_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_jt_base_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_jt_base_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「騎手・厩舎」(詳細7カテゴリ)の細分軸その1。jt(騎手・調教師の"
                      "素の成績シグナル)単体のみ。既存「騎手・厩舎 順位」(radar_jt、6シグナル"
                      "の集約)は無変更のまま残しています。",
    },
    "radar_jt_change": {
        "label": "騎手・厩舎: 乗り替わり 順位", "source": "radar_rank_jt_change",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_jt_change_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_jt_change_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_jt_change_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_jt_change_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_jt_change_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_jt_change_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「騎手・厩舎」の細分軸その2。jockey_change/prevjockey(乗り替わり"
                      "関連2シグナル)のみで構成。",
    },
    "radar_jt_stats": {
        "label": "騎手・厩舎: 掛け合わせ統計 順位", "source": "radar_rank_jt_stats",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_jt_stats_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_jt_stats_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_jt_stats_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_jt_stats_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_jt_stats_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_jt_stats_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「騎手・厩舎」の細分軸その3。odds_jockey/surf_jt/jockey_owner"
                      "(掛け合わせ統計3シグナル)のみで構成。3分割の中で最も的中率の差が"
                      "大きい軸ですが、高い的中率はodds_jockey(市場のオッズ情報)混入に"
                      "相当程度依存しており、騎手の腕そのものではなく市場評価の代理と"
                      "解釈してください。",
    },
    "radar_pedigree_pure": {
        "label": "純血統(父・母父) 順位", "source": "radar_rank_pedigree_pure",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_pedigree_pure_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_pedigree_pure_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_pedigree_pure_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_pedigree_pure_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_pedigree_pure_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_pedigree_pure_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「血統適性」(詳細7カテゴリ)の細分軸その1。sire/bms(父・母父の"
                      "適性シグナル)のみで構成。既存「血統適性 順位」(radar_pedigree、"
                      "4シグナルの集約)は無変更のまま残しています。",
    },
    "radar_pedigree_training": {
        "label": "血統×調教・コメント 順位", "source": "radar_rank_pedigree_training",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_pedigree_training_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_pedigree_training_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_pedigree_training_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_pedigree_training_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_pedigree_training_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「血統適性」の細分軸その2。ketto_training/ketto_comment(血統評価と"
                      "調教・厩舎コメントの掛け合わせ2シグナル)のみで構成。タイがやや多く"
                      "単勝的中率もベースラインと有意差が無いため、「1位のみ」は意図的に"
                      "提供していません(2026-09-15、他グループの1〜4位/1〜6位追加に合わせて"
                      "1〜2位・1〜4位・1〜6位のみ追加、この経緯は不変のため1位のみは"
                      "引き続き非提供)。",
    },
    "radar_form_agari": {
        "label": "上がり3F(近走) 順位", "source": "radar_rank_form_agari",
        "kind": "rank_le", "widget": "radio", "tier": "radar_detail",
        "options": [{"id": "radar_form_agari_top1", "label": "1位のみ", "params": {"le": 1}},
                    {"id": "radar_form_agari_top2", "label": "1〜2位", "params": {"le": 2}},
                    {"id": "radar_form_agari_top3", "label": "1〜3位", "params": {"le": 3}},
                    {"id": "radar_form_agari_top4", "label": "1〜4位", "params": {"le": 4}},
                    {"id": "radar_form_agari_top5", "label": "1〜5位", "params": {"le": 5}},
                    {"id": "radar_form_agari_top6", "label": "1〜6位", "params": {"le": 6}}],
        "extra_note": "「近走成績・調子」(詳細7カテゴリ)からagari(上がり3F)だけを"
                      "独立させた参考フィルタ。既存「近走成績・調子 順位」(radar_form、"
                      "form/margin/timediff/agari/weight_trend/class_dropの6シグナル集約)は"
                      "無変更のままです(agari抜きでも主軸のrho=0.922とほぼ不変なため、"
                      "本体を分割するのではなく本項目を外出しする形にしました)。"
                      "weight_trendは単勝的中率がベースラインと無差別だったため独立化を"
                      "見送りました(condition タイアの「馬体重の増減」で代替可能です)。",
    },
    "lap33": {
        "label": "33ラップ理論適合度", "source": "lap33_fit_rank",
        "kind": "rank_le", "widget": "radio", "tier": "reference",
        "options": [{"id": "lap33_top1", "label": "適合度1位のみ", "params": {"le": 1}},
                    {"id": "lap33_top3", "label": "適合度1〜3位", "params": {"le": 3}},
                    {"id": "lap33_top5", "label": "適合度1〜5位", "params": {"le": 5}}],
        "extra_note": "参考・不採用指標。型不明(過去走不足)の馬は対象外(NaN)。新馬戦は"
                      "ほぼ全馬が構造的に対象外になります。",
    },
    "flop": {
        "label": "凡走予測(K=3〜8参考)", "source": "flop_safety_rank",
        "kind": "rank_le", "widget": "radio", "tier": "reference",
        "options": [{"id": "flop_safe_top3", "label": "凡走しにくさ1〜3位", "params": {"le": 3}},
                    {"id": "flop_safe_top5", "label": "凡走しにくさ1〜5位", "params": {"le": 5}}],
        "extra_note": "参考・不採用指標。本番モデルには使われていません。",
    },
    "ev": {
        "label": "期待値(EV)", "source": "ev_pct",
        "kind": "threshold_ge", "widget": "radio", "tier": "reference",
        "options": [{"id": "ev_ge80", "label": "期待値80%以上", "params": {"ge": 80}},
                    {"id": "ev_ge90", "label": "期待値90%以上", "params": {"ge": 90}},
                    {"id": "ev_ge100", "label": "期待値100%以上", "params": {"ge": 100}},
                    {"id": "ev_ge120", "label": "期待値120%以上", "params": {"ge": 120}}],
        "extra_note": "参考値。既知の偏り: score比例正規化の勝率換算は人気薄を過大評価しやすい"
                      "傾向があります(2026-09-13〜: オッズはrace_results.odds_final(確定値)"
                      "優先に変更済みのため、旧来の約25%乖離バイアスは解消済み)。",
    },
    # --- 新規候補ファクター v1(2026-09-01、ユーザー依頼で追加。jra_candidate_factors_v1.py
    # で計算。既存の本番score/レーダーには一切反映していない未検証の候補群) ---
    "career_form": {
        "label": "通算複勝率(全キャリア)", "source": "career_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v1",
        "options": [{"id": "career_ge30", "label": "通算複勝率30%以上", "params": {"ge": 30}},
                    {"id": "career_ge50", "label": "通算複勝率50%以上", "params": {"ge": 50}}],
        "extra_note": "data/race_results(2026-09-13〜: 2016〜2026年に拡張、以前は2024〜2026年)"
                      "からhorse_id単位で集計、このレースより前の走のみを参照(リーク防止済み)。"
                      "新馬戦は初出走馬が大半のためNaN(対象外)になりやすい。",
    },
    "distance_band_form": {
        "label": "距離融通性(同距離帯±200m複勝率)", "source": "distance_band_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v1",
        "options": [{"id": "dist_band_ge50", "label": "同距離帯複勝率50%以上", "params": {"ge": 50}},
                    {"id": "dist_band_ge70", "label": "同距離帯複勝率70%以上", "params": {"ge": 70}}],
        "extra_note": "今回の距離±200m以内の過去走のみを対象にした複勝率。",
    },
    "course_turn_apt": {
        "label": "右回り・左回り適性(通算複勝率)", "source": "turn_apt_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v1",
        "options": [{"id": "turn_apt_ge50", "label": "同回り複勝率50%以上", "params": {"ge": 50}},
                    {"id": "turn_apt_ge70", "label": "同回り複勝率70%以上", "params": {"ge": 70}}],
        "extra_note": "東京・中京・新潟=左回り、それ以外7場=右回りとして分類。今回と同じ回りの"
                      "過去走のみを対象にした複勝率。",
    },
    "course_size_apt": {
        "label": "小回り・大箱適性(通算複勝率)", "source": "course_size_apt_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v1",
        "options": [{"id": "size_apt_ge50", "label": "同区分複勝率50%以上", "params": {"ge": 50}}],
        "extra_note": "ローカル5場(札幌・函館・福島・新潟・小倉)か中央5場(東京・中山・阪神・"
                      "京都・中京)かで区分。",
    },
    "course_slope_apt": {
        "label": "急坂コース適性(通算複勝率)", "source": "slope_apt_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v1",
        "options": [{"id": "slope_apt_ge50", "label": "急坂コース複勝率50%以上", "params": {"ge": 50}}],
        "extra_note": "中山・阪神・中京(ゴール前の急坂で知られる3場)通算複勝率。今回がこの"
                      "3場以外の場合はNaN(対象外)。",
    },
    "turf_type_apt": {
        "label": "洋芝適性(通算複勝率)", "source": "turf_type_apt_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v1",
        "options": [{"id": "turf_apt_ge50", "label": "洋芝複勝率50%以上", "params": {"ge": 50}}],
        "extra_note": "今回が札幌・函館(洋芝2場)の場合のみ評価。それ以外はNaN(対象外)。",
    },
    "grade_best": {
        "label": "重賞最高着順", "source": "grade_best",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v1",
        "options": [
            {"id": "grade_g1top3", "label": "G1で3着以内経験あり", "params": {"in": ["g1_top3"]}},
            {"id": "grade_g2top3", "label": "G2で3着以内経験あり", "params": {"in": ["g2_top3"]}},
            {"id": "grade_g3top3", "label": "G3/重賞で3着以内経験あり", "params": {"in": ["g3_top3"]}},
            {"id": "grade_no_top3", "label": "重賞出走はあるが3着内なし", "params": {"in": ["graded_no_top3"]}},
            {"id": "grade_none", "label": "重賞未出走", "params": {"in": ["no_graded_starts", "no_history"]}},
        ],
        "extra_note": "2016〜2026年の履歴内で判定(2026-09-13〜、以前は2024〜2026年。"
                      "それより前の重賞実績は反映されない)。",
    },
    "first_time_flag": {
        "label": "初サーフェス・初距離帯", "source": "first_time_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v1",
        "options": [
            {"id": "ft_surface", "label": "初サーフェス(芝/ダート初経験)", "params": {"in": ["surface_first", "both_first"]}},
            {"id": "ft_distance", "label": "初距離帯(±200m未経験)", "params": {"in": ["distance_first", "both_first"]}},
        ],
        "extra_note": "2016〜2026年の履歴範囲内での「初」判定(2026-09-13〜、以前は2024〜2026年。"
                      "それより前の経験は捕捉できない)。",
    },
    "jockey_switch": {
        "label": "乗り替わり・継続騎乗", "source": "jockey_switch",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v1",
        "options": [
            {"id": "jw_switch", "label": "乗り替わり(前走と騎手が違う)", "params": {"in": ["switch"]}},
            {"id": "jw_cont2", "label": "継続騎乗2走以上", "params": {"in": ["continuous2plus"]}},
        ],
        "extra_note": "前走(past1_jockey、フルネーム)と今回(bias_jockey、姓のみ等の短縮表記)は"
                      "書式が異なるため前方一致で同一人物判定している。同姓の別人を誤って"
                      "「継続」と判定する可能性が残る(特に多い姓の場合)。",
    },
    "trainer_change": {
        "label": "転厩初戦", "source": "trainer_change",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v1",
        "options": [{"id": "tc_changed", "label": "転厩初戦", "params": {"in": ["changed"]}}],
        "extra_note": "前走時点の調教師名(race_results、地域表記+フルネーム)と今回"
                      "(bias_trainer、姓のみ等の短縮表記)は書式が異なるため前方一致で同一人物"
                      "判定している。同姓の別人を誤って「継続」と判定する可能性が残る。",
    },
    "stable_multi_entry": {
        "label": "同一レース同厩舎多頭出し", "source": "stable_multi_entry",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v1",
        "options": [{"id": "sme_multi", "label": "同厩舎から複数出走", "params": {"in": ["multi"]}}],
    },
    "kaisai_late": {
        "label": "開催後半(荒れ馬場参考)", "source": "kaisai_late",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v1",
        "options": [{"id": "kaisai_late_yes", "label": "開催7日目以降", "params": {"in": [True]}}],
        "extra_note": "race_id構造(開催回・日目)から機械的に判定。「開催が進むほど馬場が"
                      "荒れる」という通説の検証用。",
    },
    # --- 新規候補ファクター v2(2026-09-02、ユーザー依頼で追加。jra_candidate_factors_v2.py
    # で計算。既存の本番score/レーダーには一切反映していない未検証の候補群) ---
    "class_form": {
        "label": "現クラス以上での実績", "source": "class_form_place_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v2",
        "options": [{"id": "class_form_ge30", "label": "現クラス以上複勝率30%以上", "params": {"ge": 30}},
                    {"id": "class_form_ge50", "label": "現クラス以上複勝率50%以上", "params": {"ge": 50}}],
        "extra_note": "今回と同格以上のクラスで走った過去走(2016〜2026年履歴内、2026-09-13〜、"
                      "以前は2024〜2026年、リーク防止済み)だけを対象にした複勝率。同格以上での"
                      "実走が無い馬はNaN(対象外)。",
    },
    "class_challenge": {
        "label": "格上挑戦", "source": "class_challenge_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v2",
        "options": [{"id": "class_up", "label": "自己最高クラスへの挑戦", "params": {"in": ["up_challenge"]}}],
        "extra_note": "今回のクラスが履歴内の自己最高クラスを上回る場合のみ該当。初出走はno_history"
                      "(対象外)。",
    },
    "layoff": {
        "label": "休養明け(鉄砲)", "source": "layoff_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v2",
        "options": [
            {"id": "layoff_long", "label": "半年以上(180日以上)の休み明け", "params": {"in": ["long_layoff_180d_plus"]}},
            {"id": "layoff_mid", "label": "中間休養明け(90〜179日)", "params": {"in": ["mid_layoff_90_179d"]}},
        ],
        "extra_note": "past1_date(前走日、ドット区切り書式)とkaisai_dateの日数差で判定。初出走は"
                      "no_history(対象外)。",
    },
    "second_after_layoff": {
        "label": "休み明け2走目(前走の着順は問わない)", "source": "second_after_layoff_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v2",
        "options": [{"id": "second_after_layoff_yes", "label": "該当", "params": {"in": ["second_after_layoff"]}}],
        "extra_note": "2026-09-03修正: 旧ラベル「叩き2戦目(休み明け好走後の2走目)」は実装と"
                      "不一致だったため訂正(前走(past1)自体が120日以上のブランク明けだった場合に"
                      "該当する時間差フラグのみで、前走が好走だったかは一切見ていない)。過去2走の"
                      "日付が両方揃わない馬はinsufficient_history(対象外)。",
    },
    "main_jockey_return": {
        "label": "主戦騎手への手戻り", "source": "main_jockey_return_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v2",
        "options": [
            {"id": "mj_return", "label": "主戦騎手に戻る", "params": {"in": ["return_to_main"]}},
            {"id": "mj_away", "label": "主戦騎手から離れる(乗り替わり)", "params": {"in": ["away_from_main"]}},
        ],
        "extra_note": "過去5走で最頻出(2回以上)の騎手を「主戦」と定義。past1〜5_jockey"
                      "(フルネーム)とbias_jockey(短縮表記)は書式が異なるため前方一致で同一人物"
                      "判定している(v1のjockey_switchと同じ手法、同姓の別人を誤って「同一」と"
                      "判定するリスクが残る)。",
    },
    "bad_run_pop_drop": {
        "label": "前走大敗後の人気落ち", "source": "bad_run_popularity_drop_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v2",
        "options": [{"id": "bad_run_pop_drop_yes", "label": "前走10着以下→今回7番人気以下",
                     "params": {"in": ["bad_run_then_low_pop"]}}],
        "extra_note": "前走の大敗が不利・馬場等の「言い訳できる敗戦」だったかどうかまでは判定"
                      "できない点に注意(着順と人気の組合せのみで機械的に判定)。",
    },
    "kaisai_day": {
        "label": "開催日目(生値、kaisai_lateの精緻版)", "source": "kaisai_day_num",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v2",
        "options": [
            {"id": "kaisai_day_ge5", "label": "開催5日目以降", "params": {"ge": 5}},
            {"id": "kaisai_day_ge9", "label": "開催9日目以降", "params": {"ge": 9}},
        ],
        "extra_note": "race_id構造(開催回・日目)から機械的に判定。candidate_v1のkaisai_late"
                      "(固定7日目)と違い任意の閾値で絞り込める版。",
    },
    # =====================================================================
    # 出走条件・馬プロフィール(2026-09-04追加)
    # 「予想ファクター充足度マップ」で「有(取得済み)」と判定済みなのに、これまで
    # データベース側のフィルタとして露出していなかった既存df列を列化したもの。
    # 計算はjra_candidate_factors_v3.pyで行う(本番スコアには一切反映しない)。
    # =====================================================================
    "surface": {
        "label": "サーフェス(芝/ダート)", "source": "surface",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "surface_turf", "label": "芝", "params": {"in": ["芝"]}},
            {"id": "surface_dirt", "label": "ダート", "params": {"in": ["ダ", "ダート"]}},
        ],
        "extra_note": "コース形態タイア(直線距離・高低差・周長)の各バンドは今回のサーフェスに"
                      "対応する値で判定しているため、それらと併用する場合はここも合わせて"
                      "絞り込むと解釈しやすくなります。",
    },
    "waku": {
        "label": "枠番", "source": "waku",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "waku_1_2", "label": "1〜2枠(内)", "params": {"in": [1, 2]}},
            {"id": "waku_3_6", "label": "3〜6枠(中)", "params": {"in": [3, 4, 5, 6]}},
            {"id": "waku_7_8", "label": "7〜8枠(外)", "params": {"in": [7, 8]}},
        ],
    },
    "horse_sex": {
        "label": "性別", "source": "horse_sex",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "sex_male", "label": "牡", "params": {"in": ["牡"]}},
            {"id": "sex_female", "label": "牝", "params": {"in": ["牝"]}},
            {"id": "sex_gelding", "label": "セン", "params": {"in": ["セ"]}},
        ],
        "extra_note": "newspaperのbias_sex_age(例「牝3」)を分解した値。",
    },
    "horse_age": {
        "label": "年齢", "source": "horse_age",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "age_2", "label": "2歳", "params": {"in": [2]}},
            {"id": "age_3", "label": "3歳", "params": {"in": [3]}},
            {"id": "age_4", "label": "4歳", "params": {"in": [4]}},
            {"id": "age_5", "label": "5歳", "params": {"in": [5]}},
            {"id": "age_6plus", "label": "6歳以上", "params": {"in": [6, 7, 8, 9, 10, 11, 12]}},
        ],
    },
    "weight_carried_band": {
        "label": "斤量(負担重量)", "source": "weight_carried_band",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "kin_le52", "label": "52.0kg以下(軽ハンデ)", "params": {"in": ["le52"]}},
            {"id": "kin_52_54", "label": "52.5〜54.5kg", "params": {"in": ["w52_54"]}},
            {"id": "kin_55_56", "label": "55.0〜56.5kg", "params": {"in": ["w55_56"]}},
            {"id": "kin_ge57", "label": "57.0kg以上(重ハンデ)", "params": {"in": ["ge57"]}},
        ],
    },
    "horse_weight_band": {
        "label": "馬体重(絶対値)", "source": "horse_weight_band",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "hw_small", "label": "440kg未満(小型)", "params": {"in": ["small"]}},
            {"id": "hw_medium", "label": "440〜479kg", "params": {"in": ["medium"]}},
            {"id": "hw_large", "label": "480〜519kg", "params": {"in": ["large"]}},
            {"id": "hw_xlarge", "label": "520kg以上(大型)", "params": {"in": ["xlarge"]}},
        ],
        "extra_note": "bias_horse_weight(例「418(-2)」)の絶対値部分。取得時点で当日の馬体重が"
                      "未発表だったレースが約2割あり、その馬は対象外(NaN)になります。",
    },
    "horse_weight_diff_band": {
        "label": "馬体重の増減(前走比)", "source": "horse_weight_diff_band",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "hwd_minus10", "label": "-10kg以上の大幅減", "params": {"in": ["minus10"]}},
            {"id": "hwd_minus", "label": "-1〜-9kg", "params": {"in": ["minus_small"]}},
            {"id": "hwd_zero", "label": "増減なし", "params": {"in": ["zero"]}},
            {"id": "hwd_plus", "label": "+1〜+9kg", "params": {"in": ["plus_small"]}},
            {"id": "hwd_plus10", "label": "+10kg以上の大幅増", "params": {"in": ["plus10"]}},
        ],
        "extra_note": "同じくbias_horse_weightの括弧内。初出走馬は増減表記が無いため対象外。",
    },
    "odds_band": {
        "label": "単勝オッズ帯", "source": "odds_band",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "odds_ultra_fav", "label": "1.5倍以下(断然人気)", "params": {"in": ["ultra_fav"]}},
            {"id": "odds_fav", "label": "1.6〜3.9倍", "params": {"in": ["fav"]}},
            {"id": "odds_mid", "label": "4.0〜9.9倍", "params": {"in": ["mid"]}},
            {"id": "odds_chuana", "label": "10〜29.9倍(中穴ゾーン)", "params": {"in": ["chuana"]}},
            {"id": "odds_longshot", "label": "30倍以上(大穴)", "params": {"in": ["longshot"]}},
        ],
        "extra_note": "2026-09-13〜: race_results.odds_final(確定値)優先に変更済み"
                      "(無い馬のみ取得時点値へフォールバック、期待値EVと同じ方針)。",
    },
    "interval_band": {
        "label": "レース間隔(前走からの日数)", "source": "interval_band",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "iv_renchaku", "label": "連闘(7日以内)", "params": {"in": ["renchaku"]}},
            {"id": "iv_w1_4", "label": "中1〜3週(8〜28日)", "params": {"in": ["w1_4"]}},
            {"id": "iv_w5_8", "label": "中4〜7週(29〜56日)", "params": {"in": ["w5_8"]}},
            {"id": "iv_m2_6", "label": "2〜6ヶ月(57〜180日)", "params": {"in": ["m2_6"]}},
            {"id": "iv_over6m", "label": "6ヶ月超", "params": {"in": ["over6m"]}},
        ],
        "extra_note": "past1_date(前走日)とkaisai_dateの差。前走情報が無い馬(初出走等)は"
                      "no_history(対象外)。「新規候補ファクターv2」の休養明け(layoff)"
                      "グループと母集団が一部重なります。",
    },
    "race_pace": {
        "label": "想定ペース", "source": "race_pace_label",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "pace_h", "label": "ハイ(H)", "params": {"in": ["H"]}},
            {"id": "pace_m", "label": "ミドル(M)", "params": {"in": ["M"]}},
            {"id": "pace_s", "label": "スロー(S)", "params": {"in": ["S"]}},
        ],
        "extra_note": "既存のpace_fitシグナルが算出しているレース単位のラベル(全馬同値)。",
    },
    "corner4_expected": {
        "label": "4コーナー想定位置(AI展開)", "source": "corner4_expected_rank",
        "kind": "rank_le", "widget": "radio", "tier": "condition",
        "options": [
            {"id": "c4_top1", "label": "想定先頭のみ", "params": {"le": 1}},
            {"id": "c4_top3", "label": "想定1〜3番手", "params": {"le": 3}},
            {"id": "c4_top5", "label": "想定1〜5番手", "params": {"le": 5}},
        ],
        "extra_note": "netkeibaの「AI展開」由来のcorner4_rank(発走前の予想位置であり実際の"
                      "通過順ではない)。",
    },
    "past1_finish_band": {
        "label": "前走の着順", "source": "past1_finish_band",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "p1_win", "label": "前走1着", "params": {"in": ["win"]}},
            {"id": "p1_2_3", "label": "前走2〜3着", "params": {"in": ["f2_3"]}},
            {"id": "p1_4_5", "label": "前走4〜5着(掲示板内)", "params": {"in": ["f4_5"]}},
            {"id": "p1_6_9", "label": "前走6〜9着", "params": {"in": ["f6_9"]}},
            {"id": "p1_10plus", "label": "前走10着以下", "params": {"in": ["f10plus"]}},
        ],
        "extra_note": "past1_finish。前走情報が無い馬(初出走等、本母集団で約22%)は"
                      "no_history(対象外)。",
    },
    "training_course_type": {
        "label": "最終追い切りの調教コース", "source": "training_course_type",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "tr_sakaro", "label": "坂路", "params": {"in": ["sakaro"]}},
            {"id": "tr_wood", "label": "ウッドチップ(美W・函W・CW)", "params": {"in": ["wood"]}},
            {"id": "tr_poly", "label": "ポリトラック・DP", "params": {"in": ["poly"]}},
            {"id": "tr_turf", "label": "芝", "params": {"in": ["turf"]}},
            {"id": "tr_dirt", "label": "ダート", "params": {"in": ["dirt"]}},
        ],
        "extra_note": "training_course(「栗坂」「美Ｗ」「ＣＷ」「札ダ」等)をコース種別へ分類した"
                      "もの。トレセン(美浦/栗東)か滞在先(札幌・函館)かの区別はここでは"
                      "していません。",
    },
    "training_best_time": {
        "label": "追い切りが同コース一番時計", "source": "training_best_time_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [{"id": "tr_best", "label": "一番時計", "params": {"in": ["best_time"]}}],
        "extra_note": "training_courseの末尾に「一番時計」表記が付くケース。**その日そのコースで"
                      "の最速時計**であって、その馬自身の自己ベスト更新ではありません"
                      "(自己ベスト判定には過去の調教履歴が必要で、現状は未取得)。",
    },
    "training_track_condition": {
        "label": "追い切り時の調教馬場状態", "source": "training_track_condition",
        "kind": "category_in", "widget": "checkbox", "tier": "condition",
        "options": [
            {"id": "trc_good", "label": "良", "params": {"in": ["良"]}},
            {"id": "trc_slight", "label": "稍重", "params": {"in": ["稍", "稍重"]}},
            {"id": "trc_heavy", "label": "重・不良", "params": {"in": ["重", "不", "不良"]}},
        ],
        "extra_note": "training_track_condition。重い調教馬場での好時計を評価する材料。",
    },
    "ca_jockey_win_rate": {
        "label": "コース別 騎手勝率", "source": "ca_jockey_win_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "condition",
        "options": [
            {"id": "cajw_ge10", "label": "10%以上", "params": {"ge": 10}},
            {"id": "cajw_ge15", "label": "15%以上", "params": {"ge": 15}},
            {"id": "cajw_ge20", "label": "20%以上", "params": {"ge": 20}},
        ],
        "extra_note": "netkeibaのコース分析(ca_jockey_win_rate)。今回のコース条件における"
                      "その騎手の勝率で、母数(ca_jockey_runs)が数十走と小さいケースが多く、"
                      "0%と表示される騎手が全体の約1/3を占めます。小標本ノイズに注意。",
    },
    "ca_trainer_win_rate": {
        "label": "コース別 厩舎勝率", "source": "ca_trainer_win_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "condition",
        "options": [
            {"id": "catw_ge10", "label": "10%以上", "params": {"ge": 10}},
            {"id": "catw_ge15", "label": "15%以上", "params": {"ge": 15}},
            {"id": "catw_ge20", "label": "20%以上", "params": {"ge": 20}},
        ],
        "extra_note": "同じくコース分析由来(ca_trainer_win_rate)。小標本である点は騎手側と同じ。",
    },
    # =====================================================================
    # コース形態・馬場(2026-09-04追加)
    # data/jra_course_master.csv(JRA10場の静的表)と data/jra_baba(JRA公式PDF)を結合。
    # =====================================================================
    "course_turn": {
        "label": "回り方向(右/左)", "source": "course_turn",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "turn_right", "label": "右回り", "params": {"in": ["右回り"]}},
            {"id": "turn_left", "label": "左回り", "params": {"in": ["左回り"]}},
        ],
        "extra_note": "data/jra_course_master.csv のturn_direction(東京・中京・新潟が左回り、"
                      "残り7場が右回り)。「新規候補ファクターv1」の右左回り適性(馬側の過去"
                      "複勝率)とは別物で、こちらは今回のコース属性そのものです。",
    },
    "course_turn_type": {
        "label": "内回り/外回り(コース構成)",
        "source": "course_turn_type", "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "turn_type_inner", "label": "内回り", "params": {"in": ["内回り"]}},
            {"id": "turn_type_outer", "label": "外回り", "params": {"in": ["外回り"]}},
            {"id": "turn_type_both", "label": "内外(同距離で両方施行)", "params": {"in": ["内外"]}},
            {"id": "turn_type_single", "label": "単一構成(内外の区別なし)", "params": {"in": ["単一"]}},
        ],
        "extra_note": "2026-09-06追加(改善計画1-D)。data/jra_course_turn_type.csv"
                      "(新潟・中山・京都・阪神の芝は内外複数構成、既存コースマスタの注記から"
                      "静的に整理、新規スクレイピングなし)。course_straight_bandと異なり"
                      "内外を区別できる一方、レース単位の定数のためモデルの予想スコアには"
                      "寄与しません(全馬同値の列はjra_signals._minmax()で全NaN化される構造的"
                      "制約、[[project_jra_improvement_review_2026_09_06]]参照)。ファクター"
                      "検証データベースのフィルタ、または馬側属性との交互作用の検討にのみ"
                      "使ってください。阪神芝の一部距離(1400/1600/1800/2400/2600m)は根拠と"
                      "なる公開情報が不足しているためunknown(このoptionsに未登録=常に非該当)"
                      "のままです。",
    },
    "course_straight_band": {
        "label": "直線距離", "source": "course_straight_band",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "straight_short", "label": "短い(300m未満)", "params": {"in": ["short"]}},
            {"id": "straight_mid", "label": "中(300〜450m)", "params": {"in": ["mid"]}},
            {"id": "straight_long", "label": "長い(450m以上)", "params": {"in": ["long"]}},
        ],
        "extra_note": "今回のサーフェスに対応する値(芝ならturf_straight_m、ダートなら"
                      "dirt_straight_m)で判定。コースマスタは競馬場単位の代表値であり、"
                      "新潟・京都・阪神の内回り/外回りの違いは区別できません(新潟芝は"
                      "外回り値658.7mが入っているため、内回り施行のレースでは実態より"
                      "長い側に分類されます)。",
    },
    "course_elevation_band": {
        "label": "コース高低差", "source": "course_elevation_band",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "elev_small", "label": "小(2.0m未満)", "params": {"in": ["small"]}},
            {"id": "elev_mid", "label": "中(2.0〜3.0m)", "params": {"in": ["mid"]}},
            {"id": "elev_large", "label": "大(3.0m以上)", "params": {"in": ["large"]}},
        ],
        "extra_note": "コース全体の高低差であって「ゴール前の急坂」そのものではない点に注意"
                      "(函館は高低差3.5mですが坂の位置は向正面側です)。ゴール前急坂3場"
                      "(中山・阪神・中京)を馬側の適性として見たい場合は「新規候補ファクター"
                      "v1」の急坂コース適性を使ってください。",
    },
    "course_size_band": {
        "label": "コース規模(周長)", "source": "course_size_band",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "csize_small", "label": "小回り(1700m未満)", "params": {"in": ["small"]}},
            {"id": "csize_mid", "label": "中(1700〜1900m)", "params": {"in": ["mid"]}},
            {"id": "csize_large", "label": "大箱(1900m以上)", "params": {"in": ["large"]}},
        ],
        "extra_note": "直線距離と同じくサーフェス別の周長で判定。内回り/外回りの区別は不可。",
    },
    "cushion_band": {
        "label": "クッション値(芝の硬さ)", "source": "cushion_band",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "cushion_hard", "label": "硬め(8.5未満)", "params": {"in": ["hard"]}},
            {"id": "cushion_std", "label": "標準(8.5〜9.5)", "params": {"in": ["standard"]}},
            {"id": "cushion_soft", "label": "軟らかめ(9.5以上)", "params": {"in": ["soft"]}},
        ],
        "extra_note": "JRA公式PDF(data/jra_baba)由来。重要な制約2点: (1)開催回が完全終了した"
                      "翌平日にしか公開されない事後データで、予想時点では参照できません。"
                      "(2)2025年以降の開催回しかパーサーが対応しておらず、本母集団でも"
                      "開催回PDFが未公開の70レース分は対象外(NaN)です。ダートのレースでも"
                      "同じ開催日の芝クッション値が入ります(値は芝の計測値)。",
    },
    "moisture_band": {
        "label": "含水率(ゴール前)", "source": "moisture_band",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "moist_dry", "label": "乾き気味", "params": {"in": ["dry"]}},
            {"id": "moist_std", "label": "標準", "params": {"in": ["standard"]}},
            {"id": "moist_wet", "label": "湿った", "params": {"in": ["wet"]}},
        ],
        "extra_note": "芝レースはmoisture_turf_goal_pct、ダートはmoisture_dirt_goal_pctを使用。"
                      "芝と砂で水分量のスケールが1桁違う(実測中央値: 芝13.0%・ダート3.2%)ため、"
                      "閾値はサーフェス別(芝12%/15%、ダート4%/8%)にしています。"
                      "クッション値と同じ事後データ・2025年以降のみ対応の制約があります。",
    },
    "turf_course_variant": {
        "label": "芝コース(仮柵)バリエーション", "source": "turf_course_variant",
        "kind": "category_in", "widget": "checkbox", "tier": "course_baba",
        "options": [
            {"id": "variant_a", "label": "Aコース", "params": {"in": ["A"]}},
            {"id": "variant_b", "label": "Bコース", "params": {"in": ["B"]}},
            {"id": "variant_c", "label": "Cコース", "params": {"in": ["C"]}},
            {"id": "variant_d", "label": "Dコース", "params": {"in": ["D"]}},
        ],
        "extra_note": "仮柵の移動状況。クッション値・含水率と同じPDF由来のため同じ制約"
                      "(事後データ・2025年以降のみ・未公開開催回は対象外)。本母集団では"
                      "A/Bの2種類しか出現しません。",
    },
    # =====================================================================
    # 騎手・厩舎・馬主・生産者プロフィール(2026-09-04追加、candidate_v3)
    # data/{jockey,trainer}_profile・data/horse_profile・data/race_results を結合。
    # =====================================================================
    "jockey_lead_rank": {
        "label": "騎手の年間リーディング順位", "source": "jockey_lead_rank",
        "kind": "rank_le", "widget": "radio", "tier": "candidate_v3",
        "options": [
            {"id": "jlead_top10", "label": "1〜10位", "params": {"le": 10}},
            {"id": "jlead_top30", "label": "1〜30位", "params": {"le": 30}},
            {"id": "jlead_top60", "label": "1〜60位", "params": {"le": 60}},
        ],
        "extra_note": "data/jockey_profile/{jockey_id}.csv のjra_season_rank。騎手IDは本母集団の"
                      "newspaper CSVには存在しない(bias_parser.pyがjockey_idを出力するように"
                      "なったのは2026-09-02以降の取得分から)ため、data/race_resultsの"
                      "jockey_idを(race_id, horse_id)キーで引いて結合しています。"
                      "プロフィールは取得時点のスナップショット(2026シーズン値)であり、"
                      "厳密なpoint-in-time値ではありません。",
    },
    "jockey_area": {
        "label": "騎手の所属エリア", "source": "jockey_area",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v3",
        "options": [
            {"id": "jarea_ritto", "label": "栗東(関西)", "params": {"in": ["栗東"]}},
            {"id": "jarea_miho", "label": "美浦(関東)", "params": {"in": ["美浦"]}},
            {"id": "jarea_overseas", "label": "海外(短期免許)", "params": {"in": ["海外"]}},
            {"id": "jarea_nar", "label": "地方", "params": {"in": ["地方"]}},
        ],
        "extra_note": "「海外」区分は短期免許の外国人騎手を明示的に識別できますが、"
                      "強い馬にしか騎乗依頼が来ない選抜バイアスが大きい点に注意"
                      "(充足度マップの全期間検証では勝率14.0%と全体平均の約2倍)。",
    },
    "jockey_affiliation_type": {
        "label": "騎手の所属形態", "source": "jockey_affiliation_type",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v3",
        "options": [
            {"id": "jtype_free", "label": "フリー", "params": {"in": ["free"]}},
            {"id": "jtype_stable", "label": "厩舎所属", "params": {"in": ["stable"]}},
        ],
        "extra_note": "jockey_profileのaffiliation_typeが「フリー」ならフリー、所属厩舎名が"
                      "入っていれば厩舎所属と判定。空欄(地方・海外所属の騎手に多い)は"
                      "対象外(unknown)。",
    },
    "jockey_career_wins_band": {
        "label": "騎手の通算勝利数(減量条件の代替)", "source": "jockey_career_wins_band",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v3",
        "options": [
            {"id": "jcw_le30", "label": "30勝以下(▲相当)", "params": {"in": ["le30"]}},
            {"id": "jcw_31_50", "label": "31〜50勝(△相当)", "params": {"in": ["w31_50"]}},
            {"id": "jcw_51_100", "label": "51〜100勝(☆相当)", "params": {"in": ["w51_100"]}},
            {"id": "jcw_101_500", "label": "101〜500勝", "params": {"in": ["w101_500"]}},
            {"id": "jcw_over500", "label": "501勝以上", "params": {"in": ["over500"]}},
        ],
        "extra_note": "重要: 減量ジョッキー記号(▲△☆◇)そのものは本母集団(2026-07-11〜08-23)"
                      "では取得できません。bias_jockey_allowance_markを出力するbias_parser.pyの"
                      "拡張が2026-09-02で、それ以降の新規取得分にしか付与されないためです"
                      "(実データで0件を確認済み)。代替としてJRAの減量規定と同じ区切り"
                      "(30勝以下/31〜50勝/51〜100勝)で通算勝利数をバンド化したものが本項目です。"
                      "免許取得年数の条件や女性騎手の特例は反映しておらず、また通算勝利数は"
                      "現在値スナップショットのため境界付近の騎手は実際の記号と食い違い得ます。",
    },
    "trainer_lead_rank": {
        "label": "調教師の年間リーディング順位", "source": "trainer_lead_rank",
        "kind": "rank_le", "widget": "radio", "tier": "candidate_v3",
        "options": [
            {"id": "tlead_top10", "label": "1〜10位", "params": {"le": 10}},
            {"id": "tlead_top30", "label": "1〜30位", "params": {"le": 30}},
            {"id": "tlead_top60", "label": "1〜60位", "params": {"le": 60}},
        ],
        "extra_note": "data/trainer_profile/{trainer_id}.csv のjra_season_rank。結合方法・"
                      "スナップショットである点は騎手リーディングと同じ。",
    },
    "trainer_area": {
        "label": "調教師の所属エリア", "source": "trainer_area",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v3",
        "options": [
            {"id": "tarea_ritto", "label": "栗東(関西)", "params": {"in": ["栗東"]}},
            {"id": "tarea_miho", "label": "美浦(関東)", "params": {"in": ["美浦"]}},
        ],
        "extra_note": "netkeibaの表記ゆれ(「栗東 ( 騎手成績 )」のような括弧付き接尾辞)は"
                      "除去して正規化しています。",
    },
    "transport_flag": {
        "label": "輸送競馬(所属エリアと開催場の関係)", "source": "transport_flag",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v3",
        "options": [
            {"id": "tp_home", "label": "地元開催(美浦→東京/中山、栗東→京都/阪神/中京)",
             "params": {"in": ["home"]}},
            {"id": "tp_local", "label": "ローカル遠征(札幌・函館・福島・新潟・小倉)",
             "params": {"in": ["local_away"]}},
            {"id": "tp_cross", "label": "逆エリア遠征(美浦→関西 / 栗東→関東)",
             "params": {"in": ["cross_region"]}},
        ],
        "extra_note": "調教師の所属エリアと今回の開催場から機械的に判定した簡易分類。"
                      "中京は関西圏として扱っています。実際の輸送距離・滞在の有無(ローカル"
                      "開催の滞在競馬か週中輸送か)までは判定できません。",
    },
    "owner_prior_win_rate": {
        "label": "馬主の過去勝率(このレース以前)", "source": "owner_prior_win_rate",
        "kind": "threshold_ge", "widget": "radio", "tier": "candidate_v3",
        "options": [
            {"id": "owner_ge8", "label": "馬主勝率8%以上", "params": {"ge": 8}},
            {"id": "owner_ge10", "label": "馬主勝率10%以上", "params": {"ge": 10}},
            {"id": "owner_ge12", "label": "馬主勝率12%以上", "params": {"ge": 12}},
        ],
        "extra_note": "data/race_resultsのowner_idごとに、当該レース日より前の走だけを使って"
                      "集計した勝率(リーク防止済み)。履歴が30走未満の馬主は対象外(NaN)。"
                      "履歴の始まりが2024-08-24のため、それ以前の実績は反映されません。",
    },
    "breeder_group": {
        "label": "生産牧場", "source": "breeder_group",
        "kind": "category_in", "widget": "checkbox", "tier": "candidate_v3",
        "options": [
            {"id": "breeder_northern", "label": "ノーザンファーム", "params": {"in": ["northern_farm"]}},
            {"id": "breeder_shadai", "label": "社台ファーム", "params": {"in": ["shadai_farm"]}},
            {"id": "breeder_shadai_other", "label": "その他社台グループ(白老F・追分F)",
             "params": {"in": ["shadai_group_other"]}},
            {"id": "breeder_other", "label": "その他の牧場", "params": {"in": ["other"]}},
        ],
        "extra_note": "data/horse_profile/{horse_id}.csv のbreeder。重要な制約: horse_profileは"
                      "直近14日間の出走馬2,292頭ぶんしかバックフィルされておらず(全27,152頭中の"
                      "一部)、本母集団では約44%の馬しか結合できません。残りはunknown(対象外)"
                      "で、「その他の牧場」とは区別されます。牧場別の母数が偏るため、"
                      "この項目での絞り込み結果は特に慎重に解釈してください。",
    },
}

GROUP_TIERS = [
    {"id": "basic", "label": "基本"},
    {"id": "radar", "label": "詳細7カテゴリ(レーダー)"},
    {"id": "radar_detail", "label": "レーダー細分化フィルタ(2026-09-14追加)",
     "badge": "新規追加"},
    {"id": "condition", "label": "出走条件・馬プロフィール(2026-09-04追加)",
     "badge": "新規追加"},
    {"id": "course_baba", "label": "コース形態・馬場(2026-09-04追加)",
     "badge": "新規追加"},
    {"id": "reference", "label": "参考指標(本番不採用)"},
    {"id": "candidate_v1", "label": "新規候補ファクター v1(2026-09-01追加・未検証)",
     "badge": "新規・未検証"},
    {"id": "candidate_v2", "label": "新規候補ファクター v2(2026-09-02追加・未検証)",
     "badge": "新規・未検証"},
    {"id": "candidate_v3", "label": "騎手・厩舎・馬主・生産者プロフィール"
                                   "(2026-09-04追加・未検証)",
     "badge": "新規・未検証"},
]


def evaluate_option(horse: dict, group: dict, opt: dict) -> bool:
    v = horse.get(group["source"])
    if v is None:
        return False
    kind = group["kind"]
    if kind == "rank_le":
        return v <= opt["params"]["le"]
    if kind == "threshold_ge":
        return v >= opt["params"]["ge"]
    if kind == "category_in":
        return v in opt["params"]["in"]
    raise ValueError(kind)


def horse_qualifies(horse: dict, selected_by_group: dict) -> bool:
    """selected_by_group: {group_id: set(option_id, ...)}。JS側と同じロジックを持つ
    「独立実装」として検証に使う(グループ間AND・グループ内OR)。"""
    for gid, group in FACTOR_GROUPS.items():
        selected_ids = selected_by_group.get(gid, set())
        if not selected_ids:
            continue
        opts = [o for o in group["options"] if o["id"] in selected_ids]
        if not any(evaluate_option(horse, group, o) for o in opts):
            return False
    return True
